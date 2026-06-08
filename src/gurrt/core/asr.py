import os
from pathlib import Path

import torch
from gurrt.utils.utils import audio_extraction, audio_to_text, chunk_text

def audio_extract_chunk_and_embed(
                                video_path: Path, 
                                clip_model, 
                                clip_processor, 
                                whisper_model,
                                device):
    print(f"\033[1;32mExtracting Audio\033[0m")
    audio_file = audio_extraction(path= video_path)
    print(f"\033[1;32mTranscribing Audio\033[0m")
    text = audio_to_text(audio_file, 
                         model= whisper_model,
                         beam_size= 1)
    # print(text)
    chunked_text = chunk_text(text=text)
    clip_inputs = clip_processor(text= chunked_text, 
                                 return_tensors="pt", 
                                 padding= True, 
                                 truncation = True).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()
    
    video_id = os.path.basename(video_path)
    ids = [
        f"{video_id}_chunk_{i}" 
        for i in range(len(chunked_text))
    ]
    
    metadatas = [
        {"video_path": str(video_path), "type": "audio_transcript"}
        for _ in range(len(chunked_text))
    ]
    return chunked_text, metadatas, text_features, ids