# headjack

Small Python tool for adding SQ3R-style study aids to EPUB chapters.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

By default the script uses Groq with `llama-3.3-70b-versatile`, which requires a free Groq API key:

Put your key in `.env`:

```dotenv
GROQ_API_KEY=your_groq_key
```

## Run With a File Picker

```bash
python3 main.py
```

That opens a native file picker where you can select one or more EPUB files. The tool writes new files beside the originals with `_headjack.epub` added to the filename.

EPUB page boundaries depend on the reading app and font size, so the question and reflection area is appended to the end of each chapter rather than to a fixed physical page.

You can also pass files directly:

```bash
python3 main.py book.epub another-book.epub
```

All LLM prompts live in `prompts.toml`.

All runtime settings live in `.env`. `.env.example` contains the same keys without secrets and is safe to commit.

## Groq

The app uses Groq for LLM requests:

```dotenv
GROQ_API_KEY=your_groq_key
```

Useful `.env` overrides:

```dotenv
HEADJACK_MODEL=llama-3.3-70b-versatile
HEADJACK_MAX_CHARS=50000
HEADJACK_OUTPUT_DIR=/Users/codenibler/Desktop/sq3r-output
HEADJACK_OVERWRITE=false
```

The script sends one LLM request per enriched chapter. `HEADJACK_MAX_CHARS` controls how much chapter text is sent to the model; the default is `28000`, which is meant to stay friendlier to free-tier token limits while still covering more than just the opening pages of a long chapter.
