<a href="https://pypi.org/project/gurrt/"><img src="https://img.shields.io/pypi/v/gurrt"></a>
<a href="https://pypi.org/project/gurrt/"><img src="https://img.shields.io/pypi/pyversions/gurrt"></a>
<a href="https://pypi.org/project/gurrt/"><img src="https://img.shields.io/pypi/l/gurrt"></a>
<a href="https://pepy.tech/project/gurrt"><img src="https://pepy.tech/badge/gurrt/"></a>
<a href="https://twitter.com/muffBozo"><img src="https://img.shields.io/twitter/follow/muffBozo.svg?style=social"></a>
<a href="https://x.com/as_farrr"><img src="https://img.shields.io/twitter/follow/as_farrr.svg?style=social"></a>

<p>
  <img src="https://raw.githubusercontent.com/owaismohammad/gurrt/main/gurrt.png">
</p>

<h1>gUrrT · Conversational Video Intelligence</h1>

We study a lot on YouTube and questions keep coming up mid-lecture. Existing assistants usually answer from generic world knowledge, not from the exact lecture context.

- **Google** gives broad explanations without lecture-specific grounding.
- **General LLM chat** reasons well but often has not seen your video.
- **Platform-native Q&A** can be limited to transcript-only understanding.
- **Premium video LLM flows** typically require re-uploading, have duration limits, and can be costly at scale.

The signal you need is already inside the video.

**gUrrT builds that context and lets you query it.**

## Why Existing Solutions Fall Short

Large Video Language Models (LVLMs) are capable, but expensive to run for long-form lecture understanding.

- High VRAM requirements remain common for strong open-weight video models.
- Uniform frame sampling wastes compute on near-duplicate slide frames.
- Long lectures become noisy or truncated context, reducing answer quality.

Cloud video inference can help but introduces dependency on external infrastructure, upload overhead, and usage limits.

## Where gUrrT Comes In

gUrrT shifts from **full video modeling** to **context construction + retrieval**.

- Extract only meaningful visual changes.
- Transcribe audio with Faster-Whisper.
- Embed visual/audio evidence into ChromaDB.
- Retrieve and rerank relevant context.
- Let an LLM reason over that focused evidence.

No 80 GB GPU requirement for practical lecture Q&A workflows.

```text
Video
 │
 ├── Frame Extraction (The Eyes)
 │     Temporal persistence filtering for meaningful changes
 │     ↓ Captioning backend (SmolVLM / BLIP2 / Ollama / llama.cpp path)
 │     ↓ CLIP embeddings
 │     ↓ Stored in ChromaDB
 │
 ├── Audio Pipeline (The Ears)
 │     Audio extraction + Faster-Whisper transcription
 │     ↓ Chunked + embedded
 │     ↓ Stored in ChromaDB (separate collection)
 │
 └── Query Time
       User question → query embedding → dual retrieval
       CrossEncoder reranking
       LLM synthesizes final answer
```

## v2 Direction (Context Quality First)

Answer quality follows context quality.

- **v1 failure mode A (oversampling):** too many redundant frames.
- **v1 failure mode B (undersampling):** scene-cut style detectors miss lecture slide evolution.
- **v2 direction:** temporal persistence filtering for genuine content changes.

This reduces visual noise, increases relevant coverage, and improves indexing speed by avoiding redundant captioning passes.

### Captioning & Runtime Options

| Capability | v1 | v2 |
|--|--|--|
| Captioning | BLIP-centered | SmolVLM, BLIP2, Gemma 3 via llama.cpp, Ollama |
| Audio | legacy extraction + Whisper | optimized extraction + Faster-Whisper flow |
| Interfaces | basic CLI flow | richer CLI + GUI workflow |

Backend choices:

| Backend | Command | Typical VRAM |
|---------|---------|--------------|
| SmolVLM 500M | `gurrt index <path> smolvlm` | ~4 GB |
| BLIP2 | `gurrt index <path> blip2` | ~4 GB |
| Gemma 3 via llama.cpp path | `gurrt index-llama <path>` | 4 GB+ |
| Ollama vision model | `gurrt index-ollama <path> <model>` | varies |

## Installation

Requires **Python 3.12**.

### pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install PyTorch (choose one path)
# GPU (CUDA 12.1 wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU
pip install torch torchvision torchaudio

pip install gurrt
```

### uv

```bash
pip install uv
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install gurrt
```

Optional CUDA routing in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

### clone the repo
```shell
git clone https://github.com/farehaaslam/gurrt.git
```

## How to Use

Main executable: `gurrt`

**First-time order:**

```bash
gurrt init
gurrt models-download
gurrt index ./videos/lecture.mp4 smolvlm
# then ask
gurrt ask "your question"
```

### Commands

| Command | What it does |
|---------|--------------|
| `gurrt init` | Saves Groq + Supermemory API keys. |
| `gurrt models-download` | Downloads CLIP, SmolVLM, BLIP2, Whisper, and reranker assets to local cache. |
| `gurrt index <path> smolvlm|blip2` | Indexes video frames + audio into ChromaDB. |
| `gurrt init-llama` | Prepares llama-server + Gemma artifacts for local flow. |
| `gurrt index-llama <path>` | Indexes with local llama-server captioning flow. |
| `gurrt index-ollama <path> <model>` | Indexes using selected Ollama model. |
| `gurrt ask "<query>"` | Answers using indexed context. uses groq api (Not local) |
| `gurrt chat` | Starts interactive chat session. uses llama cpp (fully local)|

> Use subcommands in the form `gurrt <command>`.

## Requirements

- Python 3.12+
- <a href="https://console.groq.com">Groq API key</a>
- <a href="https://supermemory.ai">Supermemory API key</a>
- GPU with 4 GB+ VRAM recommended (CPU fallback works but is slower)

## Project Structure

```text
gurrt/
└── src/gurrt/
    ├── api/
    │   └── server.py          # API module (experimental)
    ├── app/
    │   └── gurrt_gui.py       # Tkinter GUI
    ├── cli/
    │   └── main.py            # CLI entry point
    ├── config/
    │   └── config.py          # Config + paths
    ├── core/
    │   ├── asr.py             # Audio extraction + transcription
    │   ├── embedding.py       # Captioning + embeddings
    │   ├── llm.py             # LLM chain + Supermemory
    │   ├── models.py          # Model loading/cache/release
    │   ├── pipeline.py        # End-to-end orchestration
    │   ├── prompts.py         # Prompt templates
    │   ├── search.py          # Retrieval + reranking
    │   └── vectordb.py        # ChromaDB interface
    └── utils/
        ├── llama_server_utils.py
        └── utils.py
```

## Contributions

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your fork
5. Open a Pull Request against `main`

## License

This project is open-source under the [MIT License](LICENSE).
