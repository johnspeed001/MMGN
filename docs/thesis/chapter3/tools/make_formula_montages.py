from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
MEDIA = REPO_ROOT / "outputs" / "chapter3" / "work_inspect" / "media"
OUT = REPO_ROOT / "outputs" / "chapter3" / "work_inspect"


def trim_whitespace(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if not bbox:
        return rgb
    left, top, right, bottom = bbox
    pad = 12
    return rgb.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(rgb.width, right + pad),
            min(rgb.height, bottom + pad),
        )
    )


def make_sheet(start: int, end: int, filename: str) -> None:
    rows = []
    width = 1400
    label_width = 110
    for idx in range(start, end + 1):
        img = Image.open(MEDIA / f"image{idx}.png").convert("RGB")
        scale = min(1.0, (width - label_width - 30) / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        row_h = max(78, img.height + 24)
        row = Image.new("RGB", (width, row_h), "white")
        draw = ImageDraw.Draw(row)
        draw.text((18, 18), f"image{idx}", fill=(0, 0, 0))
        row.paste(img, (label_width, (row_h - img.height) // 2))
        draw.line((0, row_h - 1, width, row_h - 1), fill=(210, 210, 210))
        rows.append(row)
    sheet = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(OUT / filename)
    print(OUT / filename)


if __name__ == "__main__":
    from PIL import ImageChops

    make_sheet(1, 22, "formula_sheet_01_22.png")
    make_sheet(23, 42, "formula_sheet_23_42.png")
