#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = Path(__file__).with_name("prompts.toml")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Don't add SQ3R to these chapters
SKIP_WORDS = (
    "about the author",
    "acknowledgement",
    "acknowledgment",
    "afterword",
    "appendix",
    "bibliography",
    "conclusion",
    "contents",
    "copyright",
    "dedication",
    "epilogue",
    "foreword",
    "glossary",
    "index",
    "introduction",
    "notes",
    "preface",
    "prologue",
    "references",
    "title page",
    "toc",
)

STYLE = """
.ksq3r-panel {
  border: 1px solid #888;
  margin: 1.2em 0;
  padding: 0.9em;
}
.ksq3r-panel h2 {
  font-size: 1.2em;
  margin: 0 0 0.6em 0;
}
.ksq3r-link {
  display: block;
  margin: 0.8em 0;
}
.ksq3r-lines p {
  border-bottom: 1px solid #999;
  min-height: 1.7em;
  margin: 0.4em 0;
}
"""


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    api_base: str
    api_key: str | None
    prompts_path: Path
    max_chars: int
    min_chapter_chars: int
    timeout: int
    output_dir: Path | None
    overwrite: bool


@dataclass
class BookStats:
    processed: int = 0
    skipped: int = 0
    already_enriched: int = 0


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.config.provider == "ollama":
            return self._ollama(system_prompt, user_prompt)
        if self.config.provider == "openai-compatible":
            return self._openai_compatible(system_prompt, user_prompt)
        if self.config.provider == "none":
            return "LLM generation disabled for this run."
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _ollama(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.config.api_base.rstrip('/')}/api/generate",
            json={
                "model": self.config.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def _openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        base = self.config.api_base.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = requests.post(
            url,
            headers=headers,
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    epub_paths = [Path(path).expanduser() for path in args.epubs] or pick_epubs()

    if not epub_paths:
        print("No EPUB files selected.")
        return 1

    require_dependencies()
    prompts = load_prompts(config.prompts_path)
    client = LLMClient(config)

    for epub_path in epub_paths:
        output_path = choose_output_path(epub_path, config)
        stats = enrich_epub(epub_path, output_path, prompts, client, config)
        print(
            f"{epub_path.name}: enriched {stats.processed} chapter(s), "
            f"skipped {stats.skipped}, already enriched {stats.already_enriched} "
            f"-> {output_path}"
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select EPUBs and add SQ3R summary, question, and reflection blocks."
    )
    parser.add_argument("epubs", nargs="*", help="EPUB files. Omit to open a file picker.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="Path to prompts TOML.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai-compatible", "none"),
        default=os.getenv("KSQ3R_PROVIDER", "ollama"),
        help="LLM API provider. Defaults to local Ollama.",
    )
    parser.add_argument("--model", default=os.getenv("KSQ3R_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-base",
        default=os.getenv("KSQ3R_API_BASE", DEFAULT_OLLAMA_URL),
        help="Ollama base URL or OpenAI-compatible chat-completions base URL.",
    )
    parser.add_argument("--api-key", default=os.getenv("KSQ3R_API_KEY"))
    parser.add_argument("--max-chars", type=int, default=int(os.getenv("KSQ3R_MAX_CHARS", "14000")))
    parser.add_argument(
        "--min-chapter-chars",
        type=int,
        default=int(os.getenv("KSQ3R_MIN_CHAPTER_CHARS", "900")),
        help="Skip likely front/back matter and fragments shorter than this.",
    )
    parser.add_argument("--timeout", type=int, default=int(os.getenv("KSQ3R_TIMEOUT", "180")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite each source EPUB.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    api_base = args.api_base
    if args.provider == "openai-compatible" and api_base == DEFAULT_OLLAMA_URL:
        api_base = os.getenv("OPENAI_BASE_URL", api_base)
    return Config(
        provider=args.provider,
        model=args.model,
        api_base=api_base,
        api_key=args.api_key,
        prompts_path=Path(args.prompts).expanduser(),
        max_chars=args.max_chars,
        min_chapter_chars=args.min_chapter_chars,
        timeout=args.timeout,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


def pick_epubs() -> list[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilenames(
        title="Select EPUB documents",
        filetypes=(("EPUB files", "*.epub"), ("All files", "*.*")),
    )
    root.destroy()
    return [Path(path) for path in selected]


def require_dependencies() -> None:
    missing = []
    for module_name, package_name in (
        ("bs4", "beautifulsoup4"),
        ("ebooklib", "ebooklib"),
        ("requests", "requests"),
    ):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if sys.version_info < (3, 11):
        try:
            __import__("tomli")
        except ImportError:
            missing.append("tomli")
    if missing:
        joined = " ".join(missing)
        raise SystemExit(f"Missing dependencies. Install them with: python3 -m pip install {joined}")


def load_prompts(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as handle:
        prompts = tomllib.load(handle)

    for section in ("summary", "reflection"):
        for key in ("system", "user"):
            if not prompts.get(section, {}).get(key):
                raise ValueError(f"Missing [{section}].{key} in {path}")
    return prompts


def enrich_epub(
    epub_path: Path,
    output_path: Path,
    prompts: dict[str, Any],
    client: LLMClient,
    config: Config,
) -> BookStats:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    if not epub_path.exists():
        raise FileNotFoundError(epub_path)

    book = epub.read_epub(str(epub_path))
    items_by_id = {item.get_id(): item for item in book.get_items()}
    stats = BookStats()

    for spine_id in spine_item_ids(book):
        item = items_by_id.get(spine_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = ensure_body(soup)
        text = clean_text(body.get_text(" ", strip=True))
        title = chapter_title(soup, item.get_name())

        if body.find(attrs={"data-ksq3r": "summary"}):
            stats.already_enriched += 1
            continue
        if should_skip_chapter(title, item.get_name(), text, config.min_chapter_chars):
            stats.skipped += 1
            continue

        print(f"Generating SQ3R blocks for: {title}")
        chapter_text = trim_text(text, config.max_chars)
        summary = client.generate(
            prompts["summary"]["system"],
            prompts["summary"]["user"].format(title=title, text=chapter_text),
        )
        reflection = client.generate(
            prompts["reflection"]["system"],
            prompts["reflection"]["user"].format(title=title, text=chapter_text),
        )

        inject_sq3r_blocks(soup, body, title, item.get_id(), summary, reflection)
        item.set_content(str(soup).encode("utf-8"))
        stats.processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return stats


def spine_item_ids(book: Any) -> list[str]:
    ids = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        if isinstance(item_id, str) and item_id != "nav":
            ids.append(item_id)
    return ids


def ensure_body(soup: Any) -> Any:
    if soup.body:
        return soup.body
    html = soup.html or soup.new_tag("html")
    if not soup.html:
        soup.append(html)
    body = soup.new_tag("body")
    html.append(body)
    return body


def chapter_title(soup: Any, item_name: str) -> str:
    heading = soup.find(["h1", "h2", "h3", "title"])
    if heading:
        title = clean_text(heading.get_text(" ", strip=True))
        if title:
            return title
    return Path(item_name).stem.replace("_", " ").replace("-", " ").title()


def should_skip_chapter(title: str, item_name: str, text: str, min_chars: int) -> bool:
    haystack = f"{title} {item_name}".lower()
    if len(text) < min_chars:
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", haystack) for word in SKIP_WORDS)


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[Middle of chapter omitted for length]\n\n{text[-half:]}"


def inject_sq3r_blocks(
    soup: Any,
    body: Any,
    title: str,
    item_id: str,
    summary: str,
    reflection: str,
) -> None:
    safe_id = slug(item_id or title)
    start_id = f"ksq3r-start-{safe_id}"
    questions_id = f"ksq3r-questions-{safe_id}"

    add_style(soup)

    start_anchor = soup.new_tag("a", id=start_id)
    start_anchor["data-ksq3r"] = "chapter-start"

    summary_panel = panel(soup, "Chapter Preview", "summary")
    summary_panel.append(render_lines(soup, summary, ordered=False))

    jump_link = soup.new_tag("a", href=f"#{questions_id}", **{"class": "ksq3r-link"})
    jump_link.string = "Jump to the question space at the end of this chapter"

    for node in reversed((start_anchor, summary_panel, jump_link)):
        body.insert(0, node)

    questions_panel = panel(soup, "Questions", "questions")
    prompt = soup.new_tag("p")
    prompt.string = "Leave questions here before reading, then return to the beginning."
    questions_panel.append(prompt)

    lines = soup.new_tag("div", **{"class": "ksq3r-lines"})
    for _ in range(6):
        lines.append(soup.new_tag("p"))
    questions_panel.append(lines)

    reflection_panel = panel(soup, "Reflect", "reflection")
    reflection_panel.append(render_lines(soup, reflection, ordered=True))

    back_link = soup.new_tag("a", href=f"#{start_id}", **{"class": "ksq3r-link"})
    back_link.string = "Return to the beginning of this chapter"

    end_anchor = soup.new_tag("a", id=questions_id)
    end_anchor["data-ksq3r"] = "chapter-end"
    body.append(end_anchor)
    body.append(questions_panel)
    body.append(reflection_panel)
    body.append(back_link)


def add_style(soup: Any) -> None:
    head = soup.head
    if not head:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    if head.find("style", id="ksq3r-style"):
        return
    style = soup.new_tag("style", id="ksq3r-style")
    style.string = STYLE
    head.append(style)


def panel(soup: Any, title: str, kind: str) -> Any:
    section = soup.new_tag("section", **{"class": "ksq3r-panel"})
    section["data-ksq3r"] = kind
    heading = soup.new_tag("h2")
    heading.string = title
    section.append(heading)
    return section


def render_lines(soup: Any, text: str, ordered: bool) -> Any:
    lines = [strip_list_marker(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        lines = [text.strip()]

    if len(lines) == 1:
        paragraph = soup.new_tag("p")
        paragraph.string = lines[0]
        return paragraph

    list_tag = soup.new_tag("ol" if ordered else "ul")
    for line in lines:
        item = soup.new_tag("li")
        item.string = line
        list_tag.append(item)
    return list_tag


def strip_list_marker(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return clean or "chapter"


def choose_output_path(epub_path: Path, config: Config) -> Path:
    if config.overwrite:
        return epub_path

    directory = config.output_dir or epub_path.parent
    candidate = directory / f"{epub_path.stem}_sq3r{epub_path.suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{epub_path.stem}_sq3r_{counter}{epub_path.suffix}"
        counter += 1
    return candidate


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled.")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
