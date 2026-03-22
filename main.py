"""Manga translator: extract Japanese text, translate to English, erase & replace."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────────

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"

EXTRACT_PROMPT = """\
You are a manga translation expert. Analyze this manga page image carefully.

For every piece of Japanese text visible on the page (speech bubbles, narration boxes,
sound effects, signs, etc.), return a JSON array of objects with these fields:

- "original": the exact Japanese text
- "translated": a natural, contextually appropriate English translation.
  Use natural English speech patterns. Preserve tone (casual, formal, angry, etc).
- "type": one of "speech", "narration", "sfx", "sign", "other"
- "bbox": [y1, x1, y2, x2] coordinates in the range 0-1000, where 0,0 is the
  top-left corner and 1000,1000 is the bottom-right corner.
  The bbox should cover the ENTIRE writable area inside the speech bubble or text box —
  not just tightly around the characters. Include generous padding so that the full
  interior of the bubble/box is covered. This area will be erased and refilled with
  English text, so it must be large enough.

IMPORTANT:
- Return ONLY the JSON array, no markdown fences, no explanation.
- Coordinates are NORMALIZED 0-1000, not pixels.
- Include ALL visible text, even small sound effects.
- Make bounding boxes GENEROUS — cover the full interior area of bubbles/boxes.
"""


def extract_and_translate(image_path: Path) -> list[dict]:
    """Send image to Gemini, get back translated text with bounding boxes."""
    img = Image.open(image_path)
    width, height = img.size
    print(f"  Image size: {width}x{height}")

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"),
            EXTRACT_PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if Gemini wraps them anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    entries = json.loads(raw)

    # Convert normalized 0-1000 coords [y1, x1, y2, x2] to pixel coords [x1, y1, x2, y2]
    padding = 5  # extra pixels of padding
    for entry in entries:
        ny1, nx1, ny2, nx2 = entry["bbox"]
        px1 = max(0, int(nx1 / 1000 * width) - padding)
        py1 = max(0, int(ny1 / 1000 * height) - padding)
        px2 = min(width, int(nx2 / 1000 * width) + padding)
        py2 = min(height, int(ny2 / 1000 * height) + padding)
        entry["bbox"] = [px1, py1, px2, py2]

    return entries


# ── Image editing ──────────────────────────────────────────────────────────────


def find_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to find a usable font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def erase_and_replace(image_path: Path, entries: list[dict], output_path: Path):
    """Erase original text and place English translation."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for entry in entries:
        bbox = entry["bbox"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        translated = entry["translated"]
        text_type = entry.get("type", "speech")

        # Skip SFX (stylized, hard to replace cleanly) and already-English text
        if text_type == "sfx":
            continue
        if entry["original"].strip() == translated.strip():
            continue

        # Erase: fill bbox with white (works for most manga with white bubbles)
        draw.rectangle([x1, y1, x2, y2], fill="white")

        # Calculate font size to fit the box
        box_w = x2 - x1
        box_h = y2 - y1

        # Start with a size proportional to box height, then shrink to fit
        font_size = max(int(box_h * 0.3), 10)
        font = find_font(font_size)

        # Word-wrap and fit text
        lines = wrap_text(draw, translated, font, box_w - 4)

        # If text is too tall, reduce font size
        while len(lines) * (font_size + 2) > box_h and font_size > 8:
            font_size -= 1
            font = find_font(font_size)
            lines = wrap_text(draw, translated, font, box_w - 4)

        # Draw centered text
        line_height = font_size + 2
        total_text_h = len(lines) * line_height
        start_y = y1 + (box_h - total_text_h) // 2

        for i, line in enumerate(lines):
            text_bbox = draw.textbbox((0, 0), line, font=font)
            tw = text_bbox[2] - text_bbox[0]
            tx = x1 + (box_w - tw) // 2
            ty = start_y + i * line_height
            draw.text((tx, ty), line, fill="black", font=font)

    img.save(output_path, quality=95)
    print(f"  Saved: {output_path}")


def wrap_text(draw: ImageDraw.Draw, text: str, font, max_width: int) -> list[str]:
    """Simple word-wrap for text within a given pixel width."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


# ── CLI ────────────────────────────────────────────────────────────────────────


def process_page(image_path: Path, output_dir: Path):
    """Full pipeline for one page."""
    print(f"\nProcessing: {image_path.name}")

    # Step 1: Extract & translate
    entries = extract_and_translate(image_path)
    print(f"  Found {len(entries)} text regions")

    # Save translation data as JSON sidecar
    json_path = output_dir / f"{image_path.stem}_translations.json"
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    print(f"  Translations: {json_path}")

    for e in entries:
        print(f"    [{e['type']}] {e['original'][:30]}... → {e['translated'][:40]}...")

    # Step 2: Erase & replace
    output_path = output_dir / f"{image_path.stem}_en{image_path.suffix}"
    erase_and_replace(image_path, entries, output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run main.py <image_or_directory> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        process_page(input_path, output_dir)
    elif input_path.is_dir():
        pages = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        print(f"Found {len(pages)} pages in {input_path}")
        for page in pages:
            process_page(page, output_dir)
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
