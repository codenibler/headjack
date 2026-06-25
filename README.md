# headjack

Python script to generate Chapter Previews, space for noting questions you'd like answered by reading at the beginning of the chapter, available at the end of the chapter, and points to reflect on at the end.

Script is based on the [SQ3R](https://archive.umsida.ac.id/index.php/archive/preprint/view/4583) framework. In dissapointment that things like [RSVP](https://journals.sagepub.com/doi/10.1177/1529100615623267) (Rapid Serial Visual Presentation, see example [here](https://youtu.be/NdKcDPBQ-Lw?si=JEXTe8xrorVF_tFd)) and fonts like [OpenDyslexic](https://techcomm.nz/Story?Action=View&Story_id=400#:~:text=Figure%201%3A%20Notice%20the%20'heavy,way%20up%20letters%20should%20be) have no significant benefit on reading speed and can be detrimental to comprehension, I decided to ground this in what is proven to improve comprehension and retention. 

## Setup

```bash
python3 -m pip install -r requirements.txt
```

By default the script uses Groq with `llama-3.1-8b-instant`, as it has the most generous free usage limit:
Put your Groq API key in `.env`:

```dotenv
GROQ_API_KEY=your_groq_key
```

## Run With a File Picker

```bash
python3 main.py
```

Opens a native file picker where you can select one or more EPUB files. The tool writes new files beside the originals with `_headjack.epub` added to the filename.

EPUB page boundaries depend on the reading app and font size, so the question and reflection area is appended to the end of each chapter rather than to a fixed physical page.

You can also pass files directly:

```bash
python3 main.py book.epub another-book.epub
```

All LLM prompts live in `prompts.toml`, feel free to modify them to get what you want from your reading.

All runtime settings live in `.env`. Clone the `.env.example` with your own keys and preferences.  

## Parameters

You might want to change some of the default parameters in the `.env`.

```dotenv
HEADJACK_MODEL=llama-3.1-8b-instant
HEADJACK_MAX_CHARS=50000 # Truncated to beginning and end of chapter if greater than this value.
HEADJACK_OUTPUT_DIR=/Users/codenibler/Desktop/sq3r-output
HEADJACK_OVERWRITE=false
```

