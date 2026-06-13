from pathlib import Path
from platformdirs import user_config_dir
import json
import sys
class Settings:
    def __init__(self):
        home = Path(user_config_dir("gurrt"))
        home.mkdir(exist_ok=True, parents= True)
        
        config_file = home / "config.json"
        cfg = {}
        
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self.GROQ_API_KEY = cfg.get('GROQ_API_KEY')
            self.SUPERMEMORY_API_KEY = cfg.get("SUPERMEMORY_API_KEY")
        else:
            raise RuntimeError("API Keys not found")
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.LLM_MODEL="llama-3.1-8b-instant"
        self.RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.BLIP_MODEL = "Salesforce/blip-image-captioning-large"
        self.MODEL_CACHE_DIR = home / "models"
        self.CHROMA_DB_PATH= home / "chroma_db"
        self.AUDIO_PATH = home / "output.wav"
        self.LLAMA_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b4600/llama-b4600-bin-win-cuda-cu12.6-x64.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
is_windows = sys.platform == "win32"
BIN_DIR = PROJECT_ROOT / "bin"

SERVER_BIN = PROJECT_ROOT / "bin" / ("llama-server.exe" if is_windows else "llama-server")    
MODELS_DIR = PROJECT_ROOT / "models"
hf_repo = "unsloth/gemma-3-4b-it-GGUF"
model="gemma-3-4b-it-Q4_0.gguf"
mmproj_model = "mmproj-F16.gguf"
llm_path = MODELS_DIR / model
clip_path = MODELS_DIR / mmproj_model
llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
