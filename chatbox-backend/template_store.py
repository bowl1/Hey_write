import json
import os
import re
from pathlib import Path
from typing import Any

TEMPLATE_DIR = Path(os.getenv("TEMPLATE_DIR", "./templates"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "template"


def template_id_from_filename(path: Path) -> str:
    return path.stem


def is_template_file(path: Path) -> bool:
    return path.suffix == ".json" and not path.name.endswith(".embedding.json")


def infer_structure(content: str) -> list[str]:
    sections: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip().strip("#").strip("*").strip()
        if not line:
            continue
        if line.endswith(":") and len(line) <= 80:
            sections.append(line.rstrip(":").strip())
            continue
        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered and len(numbered.group(1)) <= 80:
            sections.append(numbered.group(1).strip())
    seen = set()
    unique = []
    for section in sections:
        key = section.lower()
        if key not in seen:
            unique.append(section)
            seen.add(key)
    return unique[:12]


def normalize_template(data: dict[str, Any], template_id: str | None = None) -> dict[str, Any]:
    title = str(data.get("title") or template_id or "Untitled Template").strip()
    content = str(data.get("content") or "").strip()
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not isinstance(tags, list):
        tags = []

    category = str(data.get("category") or (tags[0] if tags else "general")).strip().lower()
    description = str(data.get("description") or "").strip()
    if not description:
        description = f"Template for {title.lower()}."

    structure = data.get("structure")
    if not isinstance(structure, list):
        structure = infer_structure(content)

    return {
        "id": str(data.get("id") or template_id or slugify(title)),
        "title": title,
        "category": category,
        "description": description,
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "style": str(data.get("style") or "Formal"),
        "language": str(data.get("language") or "English"),
        "use_cases": data.get("use_cases") if isinstance(data.get("use_cases"), list) else [],
        "structure": [str(section).strip() for section in structure if str(section).strip()],
        "content": content,
        "enabled": bool(data.get("enabled", True)),
        "version": int(data.get("version") or 1),
    }


def read_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return normalize_template(data, template_id_from_filename(path))


def list_templates() -> list[dict[str, Any]]:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    templates = [
        read_template(path)
        for path in sorted(TEMPLATE_DIR.iterdir())
        if path.is_file() and is_template_file(path)
    ]
    return templates


def get_template(template_id: str) -> dict[str, Any] | None:
    for template in list_templates():
        if template["id"] == template_id:
            return template
    return None


def save_template(template: dict[str, Any]) -> dict[str, Any]:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    normalized = normalize_template(template)
    template_id = slugify(normalized["id"] or normalized["title"])
    normalized["id"] = template_id
    path = TEMPLATE_DIR / f"{template_id}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return normalized
