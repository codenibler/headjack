# Kindle SQ3R

Small Python tool for adding SQ3R-style study aids to EPUB chapters.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

By default the script uses Groq with `llama-3.3-70b-versatile`, which requires a free Groq API key:

```bash
export GROQ_API_KEY=your_groq_key
python3 main.py
```

## Run With a File Picker

```bash
python3 main.py
```

That opens a native file picker where you can select one or more EPUB files. The tool writes new files beside the originals with `_sq3r.epub` added to the filename.

EPUB page boundaries depend on the reading app and font size, so the question and reflection area is appended to the end of each chapter rather than to a fixed physical page.

You can also pass files directly:

```bash
python3 main.py book.epub another-book.epub
```

## OpenAI-Compatible APIs

For a hosted free-tier or OpenAI-compatible provider:

```bash
KSQ3R_PROVIDER=openai-compatible \
KSQ3R_API_BASE=https://example.com/v1 \
KSQ3R_API_KEY=your_key \
KSQ3R_MODEL=provider/model-name \
python3 main.py
```

All LLM prompts live in `prompts.toml`.

## Providers

Groq, recommended default:

```bash
export GROQ_API_KEY=your_groq_key
python3 main.py --provider groq
```

Hugging Face Inference Providers:

```bash
export HF_TOKEN=your_huggingface_token
python3 main.py --provider huggingface
```

Local Ollama remains available if you install it:

```bash
ollama pull llama3.2:3b
ollama serve
python3 main.py --provider ollama
```

Useful overrides:

```bash
python3 main.py --provider groq --model openai/gpt-oss-120b
python3 main.py --provider huggingface --model Qwen/Qwen2.5-7B-Instruct-1M:fastest
python3 main.py --max-chars 50000
```

The script sends one LLM request per enriched chapter. `--max-chars` controls how much chapter text is sent to the model; the default is `28000`, which is meant to stay friendlier to free-tier token limits while still covering more than just the opening pages of a long chapter.
