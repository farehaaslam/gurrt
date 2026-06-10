from pathlib import Path
import asyncio
import time
from tqdm import tqdm
import aiohttp
import torch
from gurrt.core.models import ModelManager
from gurrt.utils.utils import detect_scenes, frame_listing_uniform, scene_split, frame_listing, batched_captioning, temporal_persistence_filter, uniform_frame_sampling, uniform_frame_sampling_ollama
from gurrt.utils.llama_server_utils import batch_caption_frames,get_batch_embeddings
def scene_detection_frame_sampling(
                                   video_path: Path,
                                   clip_model, 
                                   clip_processor, 
                                   models: ModelManager,
                                #    blip_processor, 
                                #    blip_model,
                                   device):

#     scene_list = scene_split(video_path)
#     if not scene_list:
#             print("\033[1;32mSince No Scene Detected!\nFalling over to Uniform Sampling Technique\033[0m")
#             frame_PIL, timestamps_list, ids, fps = frame_listing_uniform(video_path= video_path)
            
#         #     embeddings, metadatas, ids =  uniform_frame_sampling(path = video_path,
#         #                                                          clip_model= clip_model,
#         #                                                          clip_processor= clip_processor,
#         #                                                          blip_processor=blip_processor,
#         #                                                          blip_model=blip_model,
#         #                                                          device= device)
#         #     return embeddings, metadatas, ids
#     else:
            
#         frame_PIL, timestamps_list, ids, fps = frame_listing(scene_list= scene_list, 
#                                                              video_path= video_path)
    frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path= video_path)
    blip_model, blip_processor = models.get_blip()
    caption_list, embeddings_list = batched_captioning(frame_list= frame_PIL, 
                                                       batch_size=16, 
                                                       clip_model= clip_model, 
                                                       clip_processor= clip_processor, 
                                                       blip_model= blip_model, 
                                                       blip_processor= blip_processor,
                                                       device = device)
    metadatas = [
            {
                    "caption": caption_list[i],
                    "timestamp_ms": timestamps_list[i],
                    "fps": fps,
                    "source_path": str(video_path)
            }
                for i in range(len(caption_list))
                ]
    models.release_blip()
    return embeddings_list, metadatas, ids

def scene_detection_frame_sampling_llama_server(
    video_path: Path,
    clip_model,
    clip_processor,
    device
):
    frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path=video_path)

    print(f"🎬 Dispatched {len(frame_PIL)} filtered frames to local llama-server...")
    captioned_nodes = []
    start_time = time.time()
    try:
        captioned_nodes = batch_caption_frames(frame_list=frame_PIL, concurrency_limit=4)

    except Exception as e:
        print(f"❌ Critical error during parallel batch captioning: {e}")
        return [], [], []
    end_time = time.time()
    print(f"⏱️ Batch captioning completed in {end_time - start_time:.2f} seconds. Now retrieving embeddings...")
    caption_list = [node["text"] for node in captioned_nodes]

    metadatas = [
        {
            "caption": caption_list[i],
            "timestamp_ms": timestamps_list[i],
            "fps": fps,
            "source_path": str(video_path)
        }
        for i in range(len(caption_list))
    ]
    print("batch captioning completed, now retrieving embeddings clip embeddings...")

    # --- CLIP EMBEDDING LOOP ---
    embeddings = []
    start_time = time.time()



    for i, frame in enumerate(tqdm(frame_PIL, desc="🖼️ Extracting CLIP Embeddings")):
        try:
            inputs = clip_processor(images=frame, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
            image_embedding = outputs.pooler_output          # ✅ extract tensor from wrapper object
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
            embeddings.append(image_embedding)
        except Exception as e:
            print(f"❌ CLIP embedding failed on frame {i}: {e}")
            continue
    end_time = time.time()
    print(f"⏱️ CLIP embedding extraction completed in {end_time - start_time:.2f} seconds.")
    return embeddings, metadatas, ids
  
def scene_detection_frame_sampling_ollama(
                                   video_path: Path,
                                   clip_model, 
                                   clip_processor, 
                                   model_name:str,
                                   device):

    scene_list = scene_split(video_path)
    if not scene_list:
            print("\033[1;32mSince No Scene Detected!\nFalling over to Uniform Sampling Technique\033[0m")
            embeddings, metadatas, ids =  uniform_frame_sampling_ollama(
                                                                 video_path= video_path,
                                                                 clip_model= clip_model,
                                                                 clip_processor= clip_processor,
                                                                 model_name= model_name,
                                                                 device= device)
            return embeddings, metadatas, ids
    embeddings, metadatas, ids = detect_scenes(video_path=video_path,
                                               scene_list=scene_list,
                                               clip_model=clip_model,
                                               clip_processor=clip_processor,
                                               model= model_name,
                                               device = device)
    return embeddings, metadatas, ids
