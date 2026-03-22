"""Manga translator: extract Japanese text, translate to English, erase & replace."""

import argparse
import asyncio
import json
import os
import sys
import time
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


async def extract_and_translate(image_path: Path) -> list[dict]:
    """Send image to Gemini async, get back translated text with bounding boxes."""
    img = Image.open(image_path)
    width, height = img.size

    response = await client.aio.models.generate_content(
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
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    entries = json.loads(raw)

    # Convert normalized 0-1000 coords [y1, x1, y2, x2] to pixel coords [x1, y1, x2, y2]
    padding = 5
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


def fit_text(draw: ImageDraw.Draw, text: str, box_w: int, box_h: int):
    """Find the best font size and line wrapping to fill the box well.

    Returns (font, lines, line_height, font_size).
    """
    margin = 4
    usable_w = max(box_w - margin * 2, 10)
    usable_h = max(box_h - margin * 2, 10)

    best = None

    # Try font sizes from large to small, pick the largest that fits
    max_font = min(int(usable_h * 0.8), int(usable_w * 0.6), 40)
    for font_size in range(max(max_font, 6), 5, -1):
        font = find_font(font_size)
        line_height = int(font_size * 1.2)
        lines = wrap_text(draw, text, font, usable_w)
        total_h = len(lines) * line_height

        if total_h <= usable_h:
            fits_w = all(
                draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0] <= usable_w
                for ln in lines
            )
            if fits_w:
                best = (font, lines, line_height, font_size)
                break

    # Fallback: smallest readable size
    if best is None:
        font_size = 6
        font = find_font(font_size)
        line_height = int(font_size * 1.2)
        lines = wrap_text(draw, text, font, usable_w)
        best = (font, lines, line_height, font_size)

    return best


def erase_and_replace(image_path: Path, entries: list[dict]) -> Image.Image:
    """Erase original text and place English translation. Returns the modified image."""
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

        box_w = x2 - x1
        box_h = y2 - y1

        font, lines, line_height, font_size = fit_text(draw, translated, box_w, box_h)

        # Draw centered text
        total_text_h = len(lines) * line_height
        start_y = y1 + (box_h - total_text_h) // 2

        for i, line in enumerate(lines):
            text_bbox = draw.textbbox((0, 0), line, font=font)
            tw = text_bbox[2] - text_bbox[0]
            tx = x1 + (box_w - tw) // 2
            ty = start_y + i * line_height
            draw.text((tx, ty), line, fill="black", font=font)

    return img


def wrap_text(draw: ImageDraw.Draw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap text within a given pixel width, with character-level fallback."""
    if max_width <= 0:
        return [text]

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        tw = draw.textbbox((0, 0), test, font=font)[2] - draw.textbbox((0, 0), test, font=font)[0]
        if tw <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            word_w = draw.textbbox((0, 0), word, font=font)[2] - draw.textbbox((0, 0), word, font=font)[0]
            if word_w > max_width:
                chunk = ""
                for ch in word:
                    test_ch = chunk + ch
                    cw = draw.textbbox((0, 0), test_ch, font=font)[2] - draw.textbbox((0, 0), test_ch, font=font)[0]
                    if cw <= max_width:
                        chunk = test_ch
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                current = chunk
            else:
                current = word

    if current:
        lines.append(current)
    return lines or [text]


# ── Progress tracking ─────────────────────────────────────────────────────────


class Progress:
    """Thread-safe progress tracker with CLI output."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def update(self, page_name: str, regions: int, success: bool = True):
        async with self._lock:
            if success:
                self.done += 1
            else:
                self.failed += 1
                self.done += 1

            elapsed = time.time() - self.start_time
            completed = self.done
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - completed) / rate if rate > 0 else 0

            status = "OK" if success else "FAIL"
            bar_width = 30
            filled = int(bar_width * completed / self.total)
            bar = "█" * filled + "░" * (bar_width - filled)

            sys.stderr.write(
                f"\r[{bar}] {completed}/{self.total} "
                f"({self.failed} failed) "
                f"{rate:.1f} pg/s | ETA {int(remaining)}s | "
                f"{page_name}: {regions} regions [{status}]"
                f"    "  # padding to clear previous line
            )
            sys.stderr.flush()

    def finish(self):
        elapsed = time.time() - self.start_time
        sys.stderr.write(
            f"\n\nDone: {self.done - self.failed}/{self.total} pages "
            f"({self.failed} failed) in {elapsed:.1f}s\n"
        )
        sys.stderr.flush()


# ── Concurrent processing ─────────────────────────────────────────────────────


PAGE_TIMEOUT = 120  # seconds per page before giving up


async def process_page_async(
    image_path: Path,
    semaphore: asyncio.Semaphore,
    progress: Progress,
) -> Image.Image | None:
    """Process a single page. Semaphore limits concurrency like p-queue."""
    async with semaphore:
        try:
            entries = await asyncio.wait_for(
                extract_and_translate(image_path),
                timeout=PAGE_TIMEOUT,
            )
            img = erase_and_replace(image_path, entries)
            await progress.update(image_path.name, len(entries), success=True)
            return img
        except asyncio.TimeoutError:
            await progress.update(image_path.name, 0, success=False)
            sys.stderr.write(f"\n  Timeout on {image_path.name} ({PAGE_TIMEOUT}s)\n")
            try:
                return Image.open(image_path).convert("RGB")
            except Exception:
                return None
        except Exception as e:
            await progress.update(image_path.name, 0, success=False)
            sys.stderr.write(f"\n  Error on {image_path.name}: {e}\n")
            try:
                return Image.open(image_path).convert("RGB")
            except Exception:
                return None


async def process_volume(pages: list[Path], concurrency: int, pdf_path: Path):
    """Fire all pages concurrently (like Promise.all + p-queue), save as PDF."""
    semaphore = asyncio.Semaphore(concurrency)
    progress = Progress(len(pages))

    sys.stderr.write(f"Processing {len(pages)} pages with concurrency={concurrency}\n\n")

    # Launch all at once — semaphore acts as the queue
    results = await asyncio.gather(*[
        process_page_async(page, semaphore, progress)
        for page in pages
    ])

    progress.finish()

    images = [img for img in results if img is not None]

    if not images:
        sys.stderr.write("No pages were successfully processed.\n")
        sys.exit(1)

    rgb_images = [img.convert("RGB") for img in images]
    rgb_images[0].save(pdf_path, save_all=True, append_images=rgb_images[1:])
    sys.stderr.write(f"Saved PDF: {pdf_path} ({len(images)} pages)\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Translate manga volumes to English")
    parser.add_argument("volume_folder", type=Path, help="Path to folder of manga page images")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PDF path")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Max concurrent API calls (default: 10)")
    args = parser.parse_args()

    input_path = args.volume_folder
    if not input_path.is_dir():
        print(f"Expected a directory, got: {input_path}")
        sys.exit(1)

    pages = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    if not pages:
        print(f"No .jpg or .png files found in {input_path}")
        sys.exit(1)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = args.output or output_dir / f"{input_path.name}_en.pdf"

    asyncio.run(process_volume(pages, args.concurrency, pdf_path))


if __name__ == "__main__":
    main()
