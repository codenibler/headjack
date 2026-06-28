#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = Path(os.getenv("HEADJACK_ENV_FILE", APP_DIR / ".env"))
ENV_EXAMPLE_PATH = APP_DIR / ".env.example"
PROVIDER = "groq"
MAX_COMPLETION_TOKENS = 900
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_CHAPTER_ITEM_PATTERNS = (
    r"(?:^|/)(?:endnotes?|footnotes?|notes?)(?:[_-]split[_-]?\d+)?\.x?html?$",
    r"(?:^|/)insert[_-]split[_-]?\d+\.x?html?$",
    r"(?:^|/)(?:ack|acknowledg\w*|authorbio|bibliography|copyright|cover|glossary|index|nav|toc)\w*\.x?html?$",
)
NON_CHAPTER_TITLE_PATTERNS = (
    r"\bmore praise\b",
    r"\bpraise for\b",
    r"\bcast of characters\b",
    r"\backnowledgments?\b",
)

STYLE = """
.headjack-panel {
  border: 1px solid #888;
  margin: 1.2em 0;
  padding: 0.9em;
}
.headjack-panel h2 {
  font-size: 1.2em;
  margin: 0 0 0.6em 0;
}
.headjack-link {
  display: block;
  margin: 0.8em 0;
}
.headjack-qa-block {
  margin: 0.9em 0;
}
.headjack-qa-label {
  font-weight: bold;
  margin: 0.5em 0 0.2em 0;
}
.headjack-qa-lines p {
  border-bottom: 1px solid #999;
  min-height: 1.7em;
  margin: 0.3em 0;
}
"""


@dataclass(frozen=True)
class Config:
    model: str
    api_base: str
    api_key: str | None
    prompts_path: Path
    max_chars: int
    min_chapter_chars: int
    timeout: int
    output_dir: Path | None
    overwrite: bool
    skip_words: tuple[str, ...]


@dataclass
class BookStats:
    processed: int = 0
    skipped: int = 0
    already_enriched: int = 0


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._remaining_requests: int | None = None
        self._remaining_tokens: int | None = None
        self._reset_requests_seconds: float | None = None
        self._reset_tokens_seconds: float | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        base = self.config.api_base.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": MAX_COMPLETION_TOKENS,
            "temperature": 0.2,
        }
        estimated_tokens = estimate_prompt_tokens(system_prompt, user_prompt)

        attempt = 0
        while True:
            attempt += 1
            self._wait_for_capacity(estimated_tokens)
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
            except requests.RequestException as exc:
                wait_seconds = retry_backoff_seconds(attempt)
                print(f"Groq request failed: {exc}. Retrying in {format_duration(wait_seconds)}.")
                time.sleep(wait_seconds)
                continue

            self._update_rate_limits(response.headers)
            if response.status_code in RETRYABLE_STATUS_CODES:
                wait_seconds = retry_wait_seconds(response, attempt)
                print(
                    f"Groq returned HTTP {response.status_code}{rate_limit_summary(response.headers)}"
                    f"{retryable_error_detail(response)}. "
                    f"Retrying in {format_duration(wait_seconds)}."
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code >= 400:
                raise RuntimeError(groq_error_message(response))

            self._sleep_if_nearly_limited()
            return response.json()["choices"][0]["message"]["content"].strip()

    def _update_rate_limits(self, headers: Any) -> None:
        self._remaining_requests = parse_int_header(headers.get("x-ratelimit-remaining-requests"))
        self._remaining_tokens = parse_int_header(headers.get("x-ratelimit-remaining-tokens"))
        self._reset_requests_seconds = parse_delay_header(headers.get("x-ratelimit-reset-requests"))
        self._reset_tokens_seconds = parse_delay_header(headers.get("x-ratelimit-reset-tokens"))

    def _wait_for_capacity(self, estimated_tokens: int) -> None:
        waits = []
        waiting_for_requests = False
        waiting_for_tokens = False
        if self._remaining_requests is not None and self._remaining_requests < 1:
            waits.append(self._reset_requests_seconds or 60.0)
            waiting_for_requests = True
        if self._remaining_tokens is not None and self._remaining_tokens < estimated_tokens:
            waits.append(self._reset_tokens_seconds or 60.0)
            waiting_for_tokens = True
        if waits:
            wait_seconds = max(waits) + 1
            print(f"Waiting {format_duration(wait_seconds)} for Groq rate-limit capacity.")
            time.sleep(wait_seconds)
            if waiting_for_requests:
                self._remaining_requests = None
            if waiting_for_tokens:
                self._remaining_tokens = None

    def _sleep_if_nearly_limited(self) -> None:
        waits = []
        if self._remaining_requests is not None and self._remaining_requests <= 1:
            waits.append(self._reset_requests_seconds or 60.0)
        if self._remaining_tokens is not None and self._remaining_tokens <= 1000:
            waits.append(self._reset_tokens_seconds or 60.0)
        if waits:
            wait_seconds = max(waits) + 1
            print(f"Waiting {format_duration(wait_seconds)} for Groq rate-limit reset.")
            time.sleep(wait_seconds)


def main() -> int:
    load_env(ENV_PATH)
    load_env(ENV_EXAMPLE_PATH)
    args = parse_args()
    config = build_config(args)
    validate_groq_config(config)
    require_dependencies()
    epub_paths = [Path(path).expanduser() for path in args.epubs] or pick_epubs()

    if not epub_paths:
        print("No EPUB files selected.")
        return 1

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
    parser.add_argument(
        "--prompts",
        default=setting_required("HEADJACK_PROMPTS"),
        help="Path to prompts TOML.",
    )
    parser.add_argument("--model", default=setting("HEADJACK_MODEL"))
    parser.add_argument(
        "--api-base",
        default=setting("HEADJACK_API_BASE"),
        help="Groq API base URL.",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-chars", type=int, default=setting_int("HEADJACK_MAX_CHARS"))
    parser.add_argument(
        "--min-chapter-chars",
        type=int,
        default=setting_int("HEADJACK_MIN_CHAPTER_CHARS"),
        help="Skip likely front/back matter and fragments shorter than this.",
    )
    parser.add_argument("--timeout", type=int, default=setting_int("HEADJACK_TIMEOUT"))
    parser.add_argument("--output-dir", type=Path, default=optional_path("HEADJACK_OUTPUT_DIR"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=setting_bool("HEADJACK_OVERWRITE"),
        help="Overwrite each source EPUB.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    api_base = args.api_base or groq_setting("API_BASE")
    model = args.model or groq_setting("MODEL")
    api_key = args.api_key or setting("GROQ_API_KEY")

    return Config(
        model=model,
        api_base=api_base,
        api_key=api_key,
        prompts_path=project_path(args.prompts),
        max_chars=args.max_chars,
        min_chapter_chars=args.min_chapter_chars,
        timeout=args.timeout,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        skip_words=csv_setting("HEADJACK_SKIP_WORDS"),
    )


def validate_groq_config(config: Config) -> None:
    if not config.model:
        raise SystemExit("Missing Groq model. Set HEADJACK_GROQ_MODEL.")
    if not config.api_base:
        raise SystemExit("Missing Groq API base. Set HEADJACK_GROQ_API_BASE.")
    if not config.api_key:
        raise SystemExit("Missing Groq API key. Set GROQ_API_KEY or pass --api-key.")


def estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return math.ceil((len(system_prompt) + len(user_prompt)) / 3) + MAX_COMPLETION_TOKENS


def retry_wait_seconds(response: Any, attempt: int) -> float:
    explicit_wait = parse_delay_header(response.headers.get("retry-after"))
    if explicit_wait is not None:
        return explicit_wait + 1

    reset_waits = [
        wait
        for wait in (
            parse_delay_header(response.headers.get("x-ratelimit-reset-requests")),
            parse_delay_header(response.headers.get("x-ratelimit-reset-tokens")),
            parse_delay_from_text(response.text),
        )
        if wait is not None
    ]
    if reset_waits:
        return max(reset_waits) + 1
    return retry_backoff_seconds(attempt)


def retry_backoff_seconds(attempt: int) -> float:
    return min(300.0, 2.0 ** min(attempt, 8)) + random.uniform(0.0, 1.0)


def parse_int_header(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def parse_delay_header(value: str | None) -> float | None:
    if not value:
        return None

    raw = value.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    try:
        parsed_date = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        parsed_date = None
    if parsed_date is not None:
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=dt.timezone.utc)
        return max(0.0, (parsed_date - dt.datetime.now(dt.timezone.utc)).total_seconds())

    total = 0.0
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\b", raw, flags=re.IGNORECASE))
    if not matches:
        return None
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    for match in matches:
        total += float(match.group(1)) * multipliers[match.group(2).lower()]
    return max(0.0, total)


def parse_delay_from_text(text: str) -> float | None:
    match = re.search(r"try again in\s+([0-9a-zA-Z.\s]+)", text, flags=re.IGNORECASE)
    return parse_delay_header(match.group(1)) if match else None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def rate_limit_summary(headers: Any) -> str:
    parts = []
    for label, name in (
        ("requests", "x-ratelimit-remaining-requests"),
        ("tokens", "x-ratelimit-remaining-tokens"),
    ):
        value = headers.get(name)
        if value is not None:
            parts.append(f"{label} remaining: {value}")
    return f" ({', '.join(parts)})" if parts else ""


def retryable_error_detail(response: Any) -> str:
    body = clean_text(response.text)[:300]
    return f": {body}" if body else ""


def groq_error_message(response: Any) -> str:
    body = clean_text(response.text)[:500]
    detail = f": {body}" if body else ""
    return f"Groq HTTP {response.status_code}{detail}"


def load_env(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed:
            key, value = parsed
            os.environ.setdefault(key, value)


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        value = value.split(" #", 1)[0].strip()
    return key, value


def setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip() or default


def setting_required(name: str) -> str:
    value = setting(name)
    if not value:
        raise SystemExit(f"Missing required setting {name}. Add it to .env.")
    return value


def setting_int(name: str) -> int:
    raw = setting_required(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}.") from exc


def setting_bool(name: str) -> bool:
    raw = setting_required(name).lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"{name} must be true or false, got {raw!r}.")


def csv_setting(name: str) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in setting(name).split(",") if value.strip())


def groq_setting(field: str) -> str:
    return setting(f"HEADJACK_{PROVIDER.upper()}_{field}")


def optional_path(name: str) -> Path | None:
    value = setting(name)
    return project_path(value) if value else None


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else APP_DIR / path


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

    if prompts.get("study_notes", {}).get("system") and prompts.get("study_notes", {}).get("user"):
        return prompts

    for section in ("summary", "reflection"):
        for key in ("system", "user"):
            if not prompts.get(section, {}).get(key):
                raise ValueError(f"Missing [{section}].{key} or [study_notes].{key} in {path}")
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

        if body.find(attrs={"data-headjack": "summary"}):
            stats.already_enriched += 1
            continue
        if should_skip_chapter(title, item.get_name(), text, config.min_chapter_chars, config.skip_words):
            stats.skipped += 1
            continue

        chapter_text = trim_text(text, config.max_chars)
        print(f"Generating SQ3R blocks for: {title} ({len(chapter_text):,}/{len(text):,} chars)")
        summary, reflection = generate_chapter_notes(client, prompts, title, chapter_text)

        inject_headjack_blocks(soup, body, title, item.get_id(), summary, reflection)
        item.set_content(str(soup).encode("utf-8"))
        stats.processed += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(str(output_path), book)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return stats


def generate_chapter_notes(
    client: LLMClient,
    prompts: dict[str, Any],
    title: str,
    chapter_text: str,
) -> tuple[str, str]:
    if prompts.get("study_notes"):
        for attempt in range(1, 4):
            raw_notes = client.generate(
                prompts["study_notes"]["system"],
                prompts["study_notes"]["user"].format(title=title, text=chapter_text),
            )
            try:
                notes = parse_note_json(raw_notes)
                return format_note_list(notes["summary"]), format_note_list(notes["reflection"])
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == 3:
                    print(f"Groq returned invalid JSON for {title}; falling back to separate prompts.")
                    break
                print(f"Groq returned invalid JSON for {title}: {exc}. Retrying.")

    summary = client.generate(
        prompts["summary"]["system"],
        prompts["summary"]["user"].format(title=title, text=chapter_text),
    )
    reflection = client.generate(
        prompts["reflection"]["system"],
        prompts["reflection"]["user"].format(title=title, text=chapter_text),
    )
    return summary, reflection


def parse_note_json(raw: str) -> dict[str, list[str]]:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)

    data = json.loads(extract_json_object(clean))
    return {
        "summary": normalize_note_list(data.get("summary")),
        "reflection": normalize_note_list(data.get("reflection")),
    }


def extract_json_object(text: str) -> str:
    clean = text.strip()
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return clean
    return clean[start : end + 1]


def normalize_note_list(value: Any) -> list[str]:
    if isinstance(value, list):
        lines = [clean_text(str(item)) for item in value]
    else:
        lines = [strip_list_marker(line) for line in str(value or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("The model response did not include usable study notes.")
    return lines


def format_note_list(lines: list[str]) -> str:
    return "\n".join(lines)


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


def should_skip_chapter(
    title: str,
    item_name: str,
    text: str,
    min_chars: int,
    skip_words: tuple[str, ...],
) -> bool:
    haystack = f"{title} {item_name}".lower()
    if len(text) < min_chars:
        return True
    if is_non_chapter_item(title, item_name):
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", haystack) for word in skip_words)


def is_non_chapter_item(title: str, item_name: str) -> bool:
    item_path = item_name.lower()
    title_text = title.lower()
    return any(re.search(pattern, item_path) for pattern in NON_CHAPTER_ITEM_PATTERNS) or any(
        re.search(pattern, title_text) for pattern in NON_CHAPTER_TITLE_PATTERNS
    )


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[Middle of chapter omitted for length]\n\n{text[-half:]}"


def inject_headjack_blocks(
    soup: Any,
    body: Any,
    title: str,
    item_id: str,
    summary: str,
    reflection: str,
) -> None:
    safe_id = slug(item_id or title)
    start_id = f"headjack-start-{safe_id}"
    questions_id = f"headjack-questions-{safe_id}"

    add_style(soup)

    start_anchor = soup.new_tag("a", id=start_id)
    start_anchor["data-headjack"] = "chapter-start"

    summary_panel = panel(soup, "Chapter Preview", "summary")
    summary_panel.append(render_lines(soup, summary, ordered=False))

    jump_link = soup.new_tag("a", href=f"#{questions_id}", **{"class": "headjack-link"})
    jump_link.string = "Jump to the question space at the end of this chapter"

    for node in reversed((start_anchor, summary_panel, jump_link)):
        body.insert(0, node)

    questions_panel = panel(soup, "Questions", "questions")
    prompt = soup.new_tag("p")
    prompt.string = "Use Q rows for questions before reading and A rows for answers after reading."
    questions_panel.append(prompt)

    questions_panel.append(render_question_answer_blocks(soup))

    reflection_panel = panel(soup, "Reflect", "reflection")
    reflection_panel.append(render_lines(soup, reflection, ordered=True))

    back_link = soup.new_tag("a", href=f"#{start_id}", **{"class": "headjack-link"})
    back_link.string = "Return to the beginning of this chapter"

    end_anchor = soup.new_tag("a", id=questions_id)
    end_anchor["data-headjack"] = "chapter-end"
    body.append(end_anchor)
    body.append(questions_panel)
    body.append(reflection_panel)
    body.append(back_link)


def render_question_answer_blocks(soup: Any) -> Any:
    container = soup.new_tag("div", **{"class": "headjack-qa-blocks"})
    for index in range(1, 4):
        block = soup.new_tag("div", **{"class": "headjack-qa-block"})
        block.append(qa_label(soup, f"Q{index}"))
        block.append(blank_lines(soup, 1))
        block.append(qa_label(soup, f"A{index}"))
        block.append(blank_lines(soup, 2))
        container.append(block)
    return container


def qa_label(soup: Any, label: str) -> Any:
    paragraph = soup.new_tag("p", **{"class": "headjack-qa-label"})
    paragraph.string = label
    return paragraph


def blank_lines(soup: Any, count: int) -> Any:
    lines = soup.new_tag("div", **{"class": "headjack-qa-lines"})
    for _ in range(count):
        lines.append(soup.new_tag("p"))
    return lines


def add_style(soup: Any) -> None:
    head = soup.head
    if not head:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    if head.find("style", id="headjack-style"):
        return
    style = soup.new_tag("style", id="headjack-style")
    style.string = STYLE
    head.append(style)


def panel(soup: Any, title: str, kind: str) -> Any:
    section = soup.new_tag("section", **{"class": "headjack-panel"})
    section["data-headjack"] = kind
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
    candidate = directory / f"{epub_path.stem}_headjack{epub_path.suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{epub_path.stem}_headjack_{counter}{epub_path.suffix}"
        counter += 1
    return candidate


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled.")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
