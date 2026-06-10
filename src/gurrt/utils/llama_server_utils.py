import io
import base64
import asyncio
import aiohttp
from typing import List, Dict, Any
import requests
import json
from tqdm import tqdm

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
                        "text": "Analyze this video lecture frame for a search indexing engine. Provide**On-Screen Content**: [Transcribe or summarize any key text, equations, bullet points, or diagrams visible].Be concise to the point "

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

        # progress bar tracking completions
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

        with open("captioned_nodes_debug.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

    return asyncio.run(run_pipeline())



def get_batch_embeddings(self, captions: List[str]) -> List[List[float]]:
    """
    Submits an array of text descriptions to the running local Gemma embedding server,
    utilizing hardware-level parallel execution context processing.
    """
    if not captions:
        return []
        
    url = "http://localhost:8081/v1/embeddings"
    
    # Passing the entire list array straight into the 'input' payload field
    payload = {
        "model": "embeddinggemma-300m",
        "input": captions
    }
    
    try:
        # Batch sizes take slightly longer to calculate; increase timeout window to 30s
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # Extract and return the ordered array of calculated float vectors
            # Format returned: [{"embedding": [...], "index": 0}, {"embedding": [...], "index": 1}]
            return [item["embedding"] for item in result["data"]]
        else:
            raise RuntimeError(f"Embedding engine rejected payload batch: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Failed to extract vector representation array: {e}")
        # Fallback tracking array: generate zero-filled shapes matching Gemma's 768 size limit
        return [[0.0] * 768 for _ in range(len(captions))]