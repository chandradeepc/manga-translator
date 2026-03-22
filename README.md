# manga-translator

Translate raw Japanese manga pages to English using Google Gemini. Extracts text from speech bubbles, narration boxes, and signs with bounding boxes, then erases the original Japanese and renders English translations in place.

## How it works

1. Sends manga page image to Gemini 2.5 Flash
2. Gemini returns JSON with: original text, English translation, text type, and bounding box coordinates
3. Script erases Japanese text (white fill) and renders fitted English text in the same region
4. SFX and already-English text are left untouched

## Setup

```bash
uv sync
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Usage

Single page:
```bash
uv run main.py path/to/page.jpg
```

Entire directory:
```bash
uv run main.py path/to/manga/volume/
```

Custom output directory:
```bash
uv run main.py path/to/page.jpg output/translated/
```

## Output

For each page, two files are saved to the output directory (default `./output/`):

- `{page}_en.jpg` — translated image with English text
- `{page}_translations.json` — translation data with bounding boxes

## Sample

| Original | Translated |
|----------|------------|
| Japanese speech bubbles, narration, signs | English text placed inside the same bounding regions |

## Dependencies

- [google-genai](https://pypi.org/project/google-genai/) — Gemini API client
- [Pillow](https://pypi.org/project/pillow/) — image manipulation
- [python-dotenv](https://pypi.org/project/python-dotenv/) — .env loading
