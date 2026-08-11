"""Render a reproducible terminal demo GIF from the harness's real dry-run output."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "demo.gif"
MODELS = "qwen3vl-8b-base,qwen3vl-8b-receipt-qlora,gemini-3.1-flash-lite,gpt-5.4-mini"
COMMAND = [
    sys.executable,
    "run.py",
    "--scale",
    "mini",
    "--models",
    MODELS,
    "--dry-run",
]


def _font(size: int):
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _clean_output(text: str) -> list[str]:
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    lines = [ansi.sub("", line).rstrip() for line in text.replace("\r", "").splitlines()]
    return [line for line in lines if not line.startswith("Generating ")]


def _frame(lines: list[str], visible: int) -> Image.Image:
    image = Image.new("RGB", (1200, 780), "#0b1020")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    body_font = _font(15)
    draw.rounded_rectangle(
        (18, 18, 1182, 762), radius=18, fill="#111827", outline="#334155", width=2
    )
    draw.ellipse((42, 42, 58, 58), fill="#fb7185")
    draw.ellipse((68, 42, 84, 58), fill="#fbbf24")
    draw.ellipse((94, 42, 110, 58), fill="#4ade80")
    draw.text((132, 38), "vlm-eval-bench — mini dry-run", font=title_font, fill="#e2e8f0")
    draw.line((38, 78, 1162, 78), fill="#334155", width=2)
    y = 98
    start = max(0, visible - 29)
    for index, line in enumerate(lines[start:visible], start=start):
        color = "#67e8f9" if index == 0 else "#d1d5db"
        draw.text((46, y), line[:112], font=body_font, fill=color)
        y += 22
        if y > 735:
            break
    return image


def main() -> int:
    result = subprocess.run(COMMAND, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    command_line = "$ python run.py --scale mini --models <4 configured models> --dry-run"
    lines = [command_line, *_clean_output(result.stdout)]
    stops = sorted({1, 5, 9, 14, len(lines)})
    frames = [_frame(lines, stop) for stop in stops]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=[900] * (len(frames) - 1) + [2600],
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
