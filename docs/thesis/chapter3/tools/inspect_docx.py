from __future__ import annotations

import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DOCX = ROOT / "submission.docx"
OUT_DIR = REPO_ROOT / "outputs" / "chapter3" / "work_inspect"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def text_of(element: ET.Element) -> str:
    return "".join(t.text or "" for t in element.findall(".//w:t", NS))


def rel_map(zf: zipfile.ZipFile) -> dict[str, str]:
    rel_root = ET.fromstring(zf.read("word/_rels/document.xml.rels"))
    return {r.attrib["Id"]: r.attrib["Target"] for r in rel_root}


def drawing_ids(element: ET.Element) -> list[str]:
    ids: list[str] = []
    for blip in element.findall(".//a:blip", NS):
        rid = blip.attrib.get(f"{{{NS['r']}}}embed")
        if rid:
            ids.append(rid)
    return ids


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DOCX) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
        rels = rel_map(zf)
        paras = root.findall(".//w:p", NS)
        inventory = []
        lines = []
        for idx, para in enumerate(paras):
            text = text_of(para)
            ids = drawing_ids(para)
            targets = [rels.get(rid, rid) for rid in ids]
            if text.strip() or targets:
                lines.append(f"{idx:03d}: {text}")
                if targets:
                    lines.append(f"     images: {', '.join(targets)}")
            if targets:
                inventory.append(
                    {
                        "paragraph_index": idx,
                        "text": text,
                        "images": targets,
                        "previous_text": text_of(paras[idx - 1]) if idx else "",
                        "next_text": text_of(paras[idx + 1]) if idx + 1 < len(paras) else "",
                    }
                )

        (OUT_DIR / "paragraphs.txt").write_text("\n".join(lines), encoding="utf-8")
        (OUT_DIR / "image_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        media_dir = OUT_DIR / "media"
        media_dir.mkdir(exist_ok=True)
        for name in zf.namelist():
            if name.startswith("word/media/"):
                target = media_dir / Path(name).name
                target.write_bytes(zf.read(name))

    print(f"wrote {OUT_DIR / 'paragraphs.txt'}")
    print(f"wrote {OUT_DIR / 'image_inventory.json'}")


if __name__ == "__main__":
    main()
