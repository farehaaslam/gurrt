from pathlib import Path
from gurrt.config.config import Settings
from gurrt.core.asr import audio_extract_chunk_and_embed
from gurrt.core.embedding import scene_detection_frame_sampling, scene_detection_frame_sampling_ollama,scene_detection_frame_sampling_llama_server
from gurrt.core.llm import LLMService
from gurrt.core.models import ModelManager
from gurrt.core.search import SearchService
from gurrt.core.vectordb import VectorDB
import subprocess
import requests
import time

class VideoRag:
    def __init__(self, reset:bool = False):
        self.reset =reset
        self.settings = Settings()
        self.models = ModelManager(self.settings)
        self.vectordb = VectorDB(str(self.settings.CHROMA_DB_PATH), reset=reset)
        self.llm = LLMService(self.settings)
        self.device = self.models.device
        
        self.clip_model, self.clip_processor= self.models.get_clip()
        # self.reranker = self.models.get_reranker()
        
        # self.search = SearchService(clip_model=self.clip_model,
        #                             clip_processor=self.clip_processor,
        #                             reranker= self.reranker,
        #                             vectordb= self.vectordb,
        #                             settings= self.settings)
        
    def index_video(self, video_path:Path):
        if self.reset:
            try:
                w = self.llm.delete()
                if w:
                    print("\033[1;32mSupermemory Cleared\033[0m")
                else:
                    print("\033[1;32mSupermemory Not Cleared\033[0m")
            except:
                print("\033[1;32mSupermemory Initialized✅\033[0m")
        # blip_model, blip_processor = self.models.get_blip()
        
        embeddings, metadatas, ids = scene_detection_frame_sampling(video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    # blip_model=blip_model,
                                                                    # blip_processor=blip_processor,
                                                                    models = self.models,
                                                                    device = self.device)
        self.vectordb.add_frames(ids=ids,
                                 embeddings=embeddings,
                                 metadata=metadatas)
        # self.models.release_blip()
        
        
    def index_video_ollama(self, video_path:Path, model_name: str):
        if self.reset:
            try:
                w = self.llm.delete()
                if w:
                    print("\033[1;32mSupermemory Cleared\033[0m")
                else:
                    print("\033[1;32mSupermemory Not Cleared\033[0m")
            except:
                print("\033[1;32mSupermemory Initialized✅\033[0m")
        embeddings, metadatas, ids = scene_detection_frame_sampling_ollama(
                                                                    video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    model_name=model_name,
                                                                    device= self.device)
        self.vectordb.add_frames(ids=ids,
                                 embeddings=embeddings,
                                 metadata=metadatas)
    
    def index_video_llama_server(self, video_path: Path, server_bin: Path, models_dir: Path):
        if self.reset:
            try:
                w = self.llm.delete()
                if w:
                    print("\033[1;32mSupermemory Cleared\033[0m")
                else:
                    print("\033[1;32mSupermemory Not Cleared\033[0m")
            except:
                print("\033[1;32mSupermemory Initialized✅\033[0m")
        llm_path = models_dir / "gemma-3-4b-it-Q4_0.gguf"
        clip_path = models_dir / "mmproj-model-f16.gguf"
        embed_path = models_dir / "embeddinggemma-300M-bf16.gguf"        
        cmd_caption_server = [
            str(server_bin),
            "-m", str(llm_path),
            "--mmproj", str(clip_path),
            "-ngl", "99",
            "--parallel", "4",
            "-c", "4096",
            "--port", "8080"
        ]

        # cmd_embedding_server = [
        #     str(server_bin),
        #     "-m", str(embed_path),
        #     "--embedding",
        #     "--pooling", "cls",
        #     "-ngl", "99",
        #     "-b", "1024",
        #     "-ub", "1024",
        #     "-c", "1024",
        #     "--port", "8081"
        # ]
        process_caption = None
        # process_embedding = None

        try:
            print("🔌 Launching background Gemma 3 Visual Captioning Core (Port 8080)...")
            process_caption = subprocess.Popen(
                cmd_caption_server, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )

            # print("🔌 Launching background High-Speed Gemma Embedding Core (Port 8081)...")
            # process_embedding = subprocess.Popen(
            #     cmd_embedding_server, 
            #     stdout=subprocess.DEVNULL, 
            #     stderr=subprocess.DEVNULL
            # )

            # --- SYSTEM HEALTH CHECK LAYER ---
            print("⏳ Awaiting background system engine initialization...")
            caption_ready = False
            embedding_ready = False

            # Ping both health check endpoints for up to 60 seconds
            for attempt in range(40):
                if not caption_ready:
                    try:
                        if requests.get("http://localhost:8080/health", timeout=1).status_code == 200:
                            caption_ready = True
                            print("✅ Captioning engine is online!")
                    except requests.exceptions.RequestException:
                        pass
                else:
                    break  # Exit early if captioning engine is ready    

                # if not embedding_ready:
                #     try:
                #         if requests.get("http://localhost:8081/health", timeout=1).status_code == 200:
                #             embedding_ready = True
                #             print("✅ Embedding engine is online!")
                #     except requests.exceptions.RequestException:
                #         pass

                # if caption_ready and embedding_ready:
                #     print("✅ Dual-engine core is fully online and ready!")
                #     break
                
                #time.sleep(1.5)

            if not (caption_ready ):
                raise TimeoutError("Captioning engine failed to initialize within VRAM allocation limits.")
            
            embeddings, metadatas, ids = scene_detection_frame_sampling_llama_server(
                                                                    video_path= video_path,
                                                                    clip_model=self.clip_model,
                                                                    clip_processor=self.clip_processor,
                                                                    device=self.device)
            print(f"embeddings count: {len(embeddings)}")
            print(f"metadatas count: {len(metadatas)}")
            print(f"ids count: {len(ids)}")
            if embeddings:
                print(f"first embedding length: {len(embeddings[0])}")
                print(f"first embedding type: {type(embeddings[0])}")
            self.vectordb.add_frames(ids=ids,
                                    embeddings=embeddings,
                                    metadata=metadatas)
            

        finally:
            # --- AUTOMATIC CLEANUP LAYER ---
            print("🧼 Safely terminating active background engine subprocesses...")
            
            if process_caption:
                process_caption.terminate()
                process_caption.wait()
                
            # if process_embedding:
            #     process_embedding.terminate()
            #     process_embedding.wait()
                
            print("✨ System clean up finished.")
    
    def index_audio(self, video_path:Path):
        whisper_model = self.models.get_whisper()
        chunked_text, metadatas, embeddings, ids = audio_extract_chunk_and_embed(
                                                                video_path=video_path,
                                                                clip_model=self.clip_model,
                                                                clip_processor=self.clip_processor,
                                                                whisper_model=whisper_model,
                                                                device = self.device)
        self.vectordb.add_asr(ids=ids,
                              embeddings=embeddings,
                              metadata=metadatas,
                              documents= chunked_text)
        self.models.release_whisper()
        self.models.release_all()
        
    async def ask(self, query:str):
        reranker = self.models.get_reranker()
        
        search = SearchService(clip_model=self.clip_model,
                                    clip_processor=self.clip_processor,
                                    reranker= reranker,
                                    vectordb= self.vectordb,
                                    settings= self.settings)
        caption_list, asr_list = search.query_collection(self.device,
                                                              query,
                                                              n_results=5,
                                                              )
        result = await self.llm.query_llm(query, 
                                          caption_list=caption_list, 
                                          asr_list=asr_list)
        self.models.release_all()
        return result