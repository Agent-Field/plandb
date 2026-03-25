from __future__ import annotations

from html import escape
from typing import Optional


def convert_markdown(markdown: str) -> str:
    parts: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading = _parse_heading(line)
        if heading is not None:
            level, content = heading
            parts.append(f"<h{level}>{_render_inline(content)}</h{level}>")
            continue

        parts.append(f"<p>{_render_inline(line)}</p>")

    return "\n".join(parts)


def _parse_heading(line: str) -> Optional[tuple[int, str]]:
    hashes = 0
    while hashes < len(line) and line[hashes] == "#":
        hashes += 1

    if 1 <= hashes <= 6 and len(line) > hashes and line[hashes] == " ":
        return hashes, line[hashes + 1 :].strip()

    return None


def _render_inline(text: str) -> str:
    return _render_segment(text)


def _render_segment(text: str, stop_marker: str = "") -> str:
    result: list[str] = []
    index = 0

    while index < len(text):
        if stop_marker and text.startswith(stop_marker, index):
            break

        if text.startswith("**", index):
            closing = _find_closing(text, "**", index + 2)
            if closing != -1:
                inner = _render_segment(text[index + 2 : closing], stop_marker="")
                result.append(f"<strong>{inner}</strong>")
                index = closing + 2
                continue

        if text[index] == "*":
            closing = _find_closing(text, "*", index + 1)
            if closing != -1:
                inner = _render_segment(text[index + 1 : closing], stop_marker="")
                result.append(f"<em>{inner}</em>")
                index = closing + 1
                continue

        if text[index] == "[":
            parsed_link = _parse_link(text, index)
            if parsed_link is not None:
                rendered, next_index = parsed_link
                result.append(rendered)
                index = next_index
                continue

        result.append(escape(text[index]))
        index += 1

    return "".join(result)


def _find_closing(text: str, marker: str, start: int) -> int:
    index = start
    while index < len(text):
        if text.startswith(marker, index):
            return index
        index += 1
    return -1


def _parse_link(text: str, start: int) -> Optional[tuple[str, int]]:
    label_end = text.find("]", start + 1)
    if label_end == -1 or label_end + 1 >= len(text) or text[label_end + 1] != "(":
        return None

    url_end = text.find(")", label_end + 2)
    if url_end == -1:
        return None

    label = _render_segment(text[start + 1 : label_end], stop_marker="")
    url = escape(text[label_end + 2 : url_end], quote=True)
    return f'<a href="{url}">{label}</a>', url_end + 1
