# manga-translator

Translate raw Japanese manga volumes to English using Google Gemini. Reads a folder of manga page images, extracts text from speech bubbles, narration boxes, and signs with bounding boxes, erases the original Japanese, renders English translations in place, and outputs a single PDF.

## How it works

1. Scans a volume folder for `.jpg` / `.png` page images
2. Sends pages concurrently to Gemini 2.5 Flash for OCR + translation
3. Gemini returns JSON with: original text, English translation, text type, and bounding box coordinates
4. Script erases Japanese text (white fill) and renders fitted English text in the same region
5. SFX and already-English text are left untouched
6. All translated pages are combined into a single PDF

## Setup

```bash
uv sync
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Usage

Translate a volume:
```bash
uv run main.py path/to/manga/volume/
```

Custom output path and concurrency:
```bash
uv run main.py path/to/manga/volume/ -o output/vol1_en.pdf -c 100
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `output/{folder}_en.pdf` | Output PDF path |
| `-c, --concurrency` | `10` | Max concurrent Gemini API calls |

## Output

A single PDF saved to `./output/{folder_name}_en.pdf` by default.

Progress is shown in the terminal:

```
[████████████░░░░░░░░░░░░░░░░░░] 85/227 (2 failed) 3.2 pg/s | ETA 44s | 086.jpg: 8 regions [OK]
```

## Dependencies

- [google-genai](https://pypi.org/project/google-genai/) — Gemini API client
- [Pillow](https://pypi.org/project/pillow/) — image manipulation
- [python-dotenv](https://pypi.org/project/python-dotenv/) — .env loading
