# Kindle SQ3R

Small Python tool for adding SQ3R-style study aids to EPUB chapters.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

By default the script calls a free local Ollama API:

```bash
ollama pull llama3.2:3b
ollama serve
```

## Run

```bash
python3 kindle_sq3r.py
```

That opens a native file picker where you can select one or more EPUB files. The tool writes new files beside the originals with `_sq3r.epub` added to the filename.

EPUB page boundaries depend on the reading app and font size, so the question and reflection area is appended to the end of each chapter rather than to a fixed physical page.

You can also pass files directly:

```bash
python3 kindle_sq3r.py book.epub another-book.epub
```

## OpenAI-Compatible APIs

For a hosted free-tier or OpenAI-compatible provider:

```bash
KSQ3R_PROVIDER=openai-compatible \
KSQ3R_API_BASE=https://example.com/v1 \
KSQ3R_API_KEY=your_key \
KSQ3R_MODEL=provider/model-name \
python3 kindle_sq3r.py
```

All LLM prompts live in `prompts.toml`.
