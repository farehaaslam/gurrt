import os
import logging
import sys
import time
import zipfile

import urllib
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
is_windows = sys.platform == "win32"

SERVER_BIN = PROJECT_ROOT / "bin" / ("llama-server.exe" if is_windows else "llama-server")    
MODELS_DIR = PROJECT_ROOT / "models"
logging.disable(logging.WARNING)

import typer
from pathlib import Path
from platformdirs import user_config_dir
import json
import asyncio
from gurrt.core.pipeline import VideoRag

from rich.theme import Theme
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

custom_theme = Theme({
    "primary": "bold green",
    "success": "bold bright_green",
    "error": "bold red",
    "info": "green",
    "warning": "yellow"
})

console = Console(theme= custom_theme)


app = typer.Typer(help= "🌿 gUrrT: A Video Understanding Tool")

config_dir = Path(user_config_dir("gurrt"))
config_dir.mkdir(exist_ok= True, parents= True)

@app.callback()
def main():
    title = Text("🌿 gUrrT: A Video Understanding Tool", style="bold bright_green")
    console.print(Rule(title, style="green"))
@app.command()
def init():
    """
    Initialize VideoRag by saving required API keys.
    """
    groq_link = "https://console.groq.com/docs/models"
    supermemory_link = "https://supermemory.ai/docs/integrations/supermemory-sdk"
    config_file = config_dir / "config.json"
    console.print(
        Panel(
            "[info]Get your Groq API Key here:\n[/info]"
            f"[bold green]{groq_link}[/bold green]",
            title="Groq",
            border_style="green"
        )
    )
    groq = Prompt.ask("[info]Enter Groq API Key[/info]", password=True)
    
    console.print(
        Panel(
            "[info]Get your Supermemory API Key here:\n[/info]"
            f"[bold green]{supermemory_link}[/bold green]",
            title="Supermemory",
            border_style="green"
        )
    )
    supermemory = Prompt.ask("[primary]Enter Supermemory API Key[/primary]", password=True)

    
    
    with open(config_file, "w") as f:
        json.dump({
            "GROQ_API_KEY": groq,
            "SUPERMEMORY_API_KEY": supermemory,
        }, f, indent= 2)
        
    console.print(
        Panel(
        "[success]✔ Configuration saved successfully![/success]"
        f"[success]saved at {config_file} [/success]",
        border_style= "green"
        ))

@app.command()        
@app.command()        
def init_llama():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    is_windows = sys.platform == "win32"
    
    # 1. Clean Subfolder Configuration Layout
    BIN_DIR = PROJECT_ROOT / "bin"
    SERVER_BIN = BIN_DIR / ("llama-server.exe" if is_windows else "llama-server")
    MODELS_DIR = PROJECT_ROOT / "models"
    
    llm_path = MODELS_DIR / "gemma-3-4b-it-Q4_0.gguf"
    clip_path = MODELS_DIR / "mmproj-model-f16.gguf"
    #embed_path = MODELS_DIR / "embeddinggemma-300M-bf16.gguf"
    
    if not llm_path.exists() or not clip_path.exists() :
        console.print("[error]❌ Error: Fixed GGUF model components missing from the root /models/ folder.[/error]")
        console.print("[warning]Please run this setup downloader command first:[/warning]\n👉 [bold cyan]gurrt models-download[/bold cyan]\n")
        raise typer.Exit(code=1)
   
    # Check if the subfolder already contains our structural setup
    if SERVER_BIN.exists() and (BIN_DIR / "llama.dll").exists():
        console.print("[success]✔ Isolated runtime server binaries and dependencies verified.[/success]")
        return
        
    # Ensure our isolated binary subdirectory exists
    BIN_DIR.mkdir(parents=True, exist_ok=True)
        
    console.print("[warning]⚠️ Runtime dependencies missing. Fetching latest release assets via GitHub API...[/warning]")
    try:
        # Fetch release metadata metadata safely
        api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req) as response:
            release_data = json.loads(response.read().decode())
        
        download_url = None
        for asset in release_data.get("assets", []):
            name = asset.get("name", "").lower()
            if "bin-win" in name and "cuda" in name and "cudart" not in name and name.endswith(".zip"):
                download_url = asset.get("browser_download_url")
                break
                
        if not download_url:
            for asset in release_data.get("assets", []):
                name = asset.get("name", "").lower()
                if "bin-win" in name and "cpu" in name and name.endswith(".zip"):
                    download_url = asset.get("browser_download_url")
                    break
        
        zip_path = PROJECT_ROOT / "temp_server.zip"
        urllib.request.urlretrieve(download_url, zip_path)
        
        # Extract target engine components directly into the isolated bin directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            archive_files = zip_ref.namelist()
            extracted_count = 0
            
            for file_path in archive_files:
                filename = os.path.basename(file_path)
                lowered_filename = filename.lower()
                
                if not filename:
                    continue
                
                # Check for the primary server executable shell
                if lowered_filename == "llama-server.exe" or lowered_filename == "llama-server":
                    member_data = zip_ref.read(file_path)
                    with open(SERVER_BIN, "wb") as target_file:
                        target_file.write(member_data)
                    extracted_count += 1
                
                # Grab ALL essential core DLLs and explicit architectural fallbacks
                elif lowered_filename.endswith(".dll") and ("llama" in lowered_filename or "ggml" in lowered_filename or "cublas" in lowered_filename or "cudart" in lowered_filename):
                    member_data = zip_ref.read(file_path)
                    dll_target_path = BIN_DIR / filename  # Target our subfolder path explicitly
                    with open(dll_target_path, "wb") as target_file:
                        target_file.write(member_data)
                    extracted_count += 1
            
            if extracted_count == 0:
                raise FileNotFoundError("Could not find required execution components in this archive.")
                
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        console.print("[success]✔ Runtime server engine successfully isolated inside /bin/ directory![/success]")
        
    except Exception as e:
        console.print(f"[error]❌ Automation failed to retrieve server asset: {e}[/error]")
        if 'zip_path' in locals() and os.path.exists(zip_path):
            os.remove(zip_path)
        raise typer.Exit(code=1) 
@app.command()
def index_llama(video_path):
    """
    Index a video using local llama-server for captioning and embedding.
    """
    console.print(
        Panel(
            f"[primary]Indexing Video with Local Llama-Server[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    
    video_time_start = time.time()
    rag.index_video_llama_server(video_path=video_path, server_bin=SERVER_BIN, models_dir=MODELS_DIR)
    video_time_end = time.time()
    
    audio_time_start = time.time()
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    audio_time_end = time.time()
    
    console.print(Panel(
            "[success]✔ Video indexed successfully using local Llama-Server![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    print(f"Video Indexing Time: {video_time_end - video_time_start:.2f} seconds")
    print(f"Audio Indexing Time: {audio_time_end - audio_time_start:.2f} seconds")


@app.command()
def models_download():
    """
    Download and cache all required AI models locally.
    """
    cache_dir = config_dir /"models"
    cache_dir.mkdir(exist_ok= True, parents= True)

    console.print(
        Panel(
            "[primary]Downloading Models[/primary]",
            border_style="green"
        )
    )
    from gurrt.core.models import download_models
    with Progress(
        SpinnerColumn(style="green"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="green"),
        console=console,
    ) as progress:
        task = progress.add_task("[info]Downloading models...", total=100)
        download_models(cache_dir)
        progress.update(task, completed=100)

    console.print(f"[success]✔ Models cached successfully at {cache_dir}![/success]")



@app.command()
def index(video_path):
    """
    Index a video by extracting frames and audio for retrieval.
    """
    console.print(
        Panel(
            f"[primary]Indexing Video[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    
    video_time_start = time.time()
    rag.index_video(video_path=video_path)
    video_time_end = time.time()
    
    audio_time_start = time.time()
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    audio_time_end = time.time()
    
    console.print(Panel(
            "[success]✔ Video indexed successfully![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    print(f"Video Indexing Time: {video_time_end - video_time_start:.2f} seconds")
    print(f"Audio Indexing Time: {audio_time_end - audio_time_start:.2f} seconds")

@app.command()
def index_ollama(video_path, model_name):
    """
    Index a video by extracting frames and audio for retrieval with Ollama Models.
    Plug in your Ollama Model Name
    """
    console.print(
        Panel(
            f"[primary]Indexing Video With Ollama[/primary]\n[info]{video_path}[/info]",
            border_style="green"
        )
    )
    rag = VideoRag(reset=True)
    rag.index_video_ollama(video_path=video_path, model_name= model_name)
    
    with console.status("[info]Processing audio transcription...[/info]", spinner="dots"):
        rag.index_audio(video_path=video_path)
    
    console.print(Panel(
            "[success]✔ Video indexed successfully![/success]"
            "[success]You may start asking your queries![/success]",
            border_style="green"
        ))
    
@app.command(help = "Ask a question about an indexed video.")
def ask(query:str):
    """
    Ask a question about an indexed video.
    """
    rag = VideoRag()
    
    with console.status("[info]Thinking...[/info]", spinner="dots"):
        response = asyncio.run(rag.ask(query= query))
    console.print(
        Panel(
            response,
            title="[success]Answer[/success]",
            border_style="green"
        )
    )
        
if __name__ == "__main__":
    app()