"""ATX-only Markdown section parsing.

Supports headings written in ATX style: one to six leading ``#`` characters,
a required space (or nothing after them at all), and an optional closing
run of ``#`` characters. Setext headings (a line of text underlined with
``=`` or ``-``) are intentionally not recognized as section boundaries: a
small deterministic line scanner has no reliable way to tell an underline
apart from a horizontal rule or a plain paragraph without much more context
than this parser tracks, and CommonMark itself treats that construct as one
of the more ambiguous parts of the spec. This is not a CommonMark
implementation; only what is needed to keep engineering-doc headings honest
is here.

Fenced code blocks (opened with three or more backticks or tildes) are
tracked with a small state machine so a ``#``-prefixed line inside one is
never mistaken for a heading.
"""

from __future__ import annotations

import re

from engineering_knowledge.domain import SectionPath

_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})(?:[ \t]+(.*))?$")
_ATX_TRAILING_HASHES_PATTERN = re.compile(r"(?:^|[ \t])#+[ \t]*$")
_FENCE_OPEN_PATTERN = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def split_into_sections(normalized_text: str) -> list[tuple[SectionPath, str]]:
    """Split normalized Markdown into (section_path, section_text) pairs, in order.

    The heading line that introduces a section belongs to that section, not
    to the one before it. A section with no content (no preamble before the
    first heading, for instance) is never emitted: an entry only appears
    here if there is real source text behind it, even if that text is just
    the heading line itself.

    Concatenating every returned section_text in order reproduces
    ``normalized_text`` exactly: sections partition the input, they never
    drop or rewrite any of it.
    """
    sections: list[tuple[SectionPath, str]] = []
    # Tracked as (level, text) pairs, not just text: truncating on a new
    # heading has to compare actual heading levels, not stack position.
    # Those two only coincide when every level from 1 upward has actually
    # appeared, which a document with skipped or non-monotonic levels (a
    # level-2 heading with no level-1 parent, for instance) breaks.
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    in_fence = False
    fence_char = ""
    fence_length = 0

    def flush_section() -> None:
        if current_lines:
            headings = tuple(text for _, text in heading_stack)
            sections.append((SectionPath(headings=headings), "".join(current_lines)))
            current_lines.clear()

    for line in normalized_text.splitlines(keepends=True):
        stripped_line = line.rstrip("\n")

        if in_fence:
            current_lines.append(line)
            if _is_fence_close(stripped_line, fence_char, fence_length):
                in_fence = False
            continue

        fence_open = _match_fence_open(stripped_line)
        if fence_open is not None:
            current_lines.append(line)
            fence_char, fence_length = fence_open
            in_fence = True
            continue

        heading = _parse_atx_heading(stripped_line)
        if heading is not None:
            flush_section()
            level, heading_text = heading
            heading_stack = [entry for entry in heading_stack if entry[0] < level]
            heading_stack.append((level, heading_text))
            current_lines.append(line)
            continue

        current_lines.append(line)

    flush_section()
    return sections


def _match_fence_open(line: str) -> tuple[str, int] | None:
    match = _FENCE_OPEN_PATTERN.match(line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _is_fence_close(line: str, fence_char: str, fence_length: int) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(char != fence_char for char in stripped):
        return False
    return len(stripped) >= fence_length


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    """Parse an ATX heading line, or report that this line is not one after all.

    A line that looks structurally like an ATX heading (``#``, ``##``, a
    bare ``###``) but whose canonical title strips down to nothing, for
    example ``#`` alone or ``### ###``, is not treated as a heading at all.
    Source documents are untrusted content, and a document containing a
    line like that is not malformed on its own terms, it just has no
    section to introduce; the caller falls back to treating the whole line
    as ordinary content of whatever section is already open.
    """
    match = _ATX_HEADING_PATTERN.match(line)
    if match is None:
        return None

    level = len(match.group(1))
    text = (match.group(2) or "").strip()

    trailing = _ATX_TRAILING_HASHES_PATTERN.search(text)
    if trailing is not None:
        text = text[: trailing.start()].rstrip()

    if not text:
        return None

    return level, text
