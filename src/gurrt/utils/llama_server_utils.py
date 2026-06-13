import io
import base64
import asyncio
import time
import aiohttp
from typing import List, Dict, Any
import requests
import json
from tqdm import tqdm
from gurrt.utils.utils import temporal_persistence_filter
from pathlib import Path
from huggingface_hub import hf_hub_download
from  gurrt.config.config import LlamaServerManager
from gurrt.core.prompts import GEMMA_CAPTION_PROMPT



def _convert_pil_to_base64(pil_img) -> str:
    """Converts a PIL image object to a base64 string completely in memory."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def _caption_single_frame_worker(
    session: aiohttp.ClientSession, 
    b64_image: str, 
    index: int, 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Sends an individual base64 string to the running local Gemma 3 engine."""
    server_url = "http://localhost:8080/v1/chat/completions"
    
    payload = {
        "model": "gemma-3-4b-it", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text":GEMMA_CAPTION_PROMPT

                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ],
        "temperature": 0.1
    }
    
    async with semaphore:
        try:
            async with session.post(server_url, json=payload, timeout=45) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    caption = result["choices"][0]["message"]["content"]
                    return {"index": index, "text": caption}
                else:
                    print(f"⚠️ Engine error on node index {index}: Status Code {resp.status}")
                    return {"index": index, "text": "Error: Failed to generate description."}
        except Exception as e:
            print(f"❌ Server timeout or network loss on node index {index}: {e}")
            return {"index": index, "text": "Error: Pipeline connection exception."}

def batch_caption_frames(frame_list: list, concurrency_limit: int = 4) -> List[Dict[str, Any]]:
    async def run_pipeline():
        semaphore = asyncio.Semaphore(concurrency_limit)
        tasks = []
        completed = 0
        total = len(frame_list)
        pbar = tqdm(total=total, desc="🧠 Captioning Frames", unit="frame")

        async def tracked_worker(session, b64_str, idx, semaphore):
            nonlocal completed
            result = await _caption_single_frame_worker(session, b64_str, idx, semaphore)
            completed += 1
            active = len([t for t in tasks if not t.done()])
            pbar.set_postfix({
                "active": min(active, concurrency_limit),  # concurrent slots in use
                "done": completed,
                "queued": total - completed
            })
            pbar.update(1)
            return result

        async with aiohttp.ClientSession() as session:
            for idx, pil_frame in enumerate(frame_list):
                try:
                    b64_str = _convert_pil_to_base64(pil_frame)
                    task = asyncio.create_task(
                        tracked_worker(session, b64_str, idx, semaphore)
                    )
                    tasks.append(task)
                except Exception as e:
                    print(f"Skipping corrupt frame at index {idx}: {e}")

            print(f"📦 Dispatched {len(tasks)} tasks | concurrency limit: {concurrency_limit}")
            results = await asyncio.gather(*tasks)

        pbar.close()
        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x["index"])

        # with open("captioned_nodes_debug.json", "w") as f:
        #     json.dump(results, f, indent=4)

        return results

    return asyncio.run(run_pipeline())

def wait_for_server():
    print("⏳ Awaiting background system engine initialization...")
    for attempt in range(40):
        try:
            if requests.get("http://localhost:8080/health", timeout=1).status_code == 200:
                print("✅ Captioning engine is online!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.5)  
    return False    

def process_video(video_path):
    print("🎬 Starting video temporal persistence filtering...")
    return temporal_persistence_filter(video_path=video_path)


def download_gemma3_models(models_dir: Path):
    """
    Sequentially downloads Gemma 3 model weights and its associated 
    multimodal vision projector from Hugging Face Hub.
    """

    models_dir.mkdir(exist_ok=True, parents=True)
    #enable_progress_bars()  
    llama_server_manager = LlamaServerManager()
    huggingface_repo = llama_server_manager.hf_repo
    files = [
        llama_server_manager.model_filename, 
        llama_server_manager.mmproj_filename
    ]

    for filename in files:
        target_path = models_dir / filename
        
        if not target_path.exists():
            print(f"Downloading {filename} from Hugging Face...")
            hf_hub_download(
                repo_id=huggingface_repo,
                filename=filename,
                local_dir=str(models_dir),
            )
            print(f"✔ Successfully acquired {filename}")
        else:
            print(f"✔ {filename} already exists in target directory. Skipping.")