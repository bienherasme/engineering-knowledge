"""Bounded, deterministic chunking of normalized document text.

Two small entry points, ``chunk_markdown`` and ``chunk_plain_text``, cover
the only two source formats v0 supports. Both normalize their input, split
it into (section_path, section_text) pairs (Markdown parses real section
structure; plain text is a single root section), then split each section's
text into bounded pieces and hand back fully constructed ``Chunk`` domain
models.

No overlap between chunks in v0.1.0: overlap duplicates source text and
complicates both provenance and identity, and should be justified by
retrieval evaluation results rather than assumed up front.

Splitting always prefers a natural boundary over a mechanical one: whole
paragraphs are packed together first, falling back to line boundaries when
a paragraph alone exceeds the bound, and falling back to a hard character
split only for a single line that alone exceeds the bound. Because every
tier is just a different granularity of partitioning the same character
stream, concatenating a section's chunks in order reproduces that section's
text exactly, no source content is ever dropped or rewritten.
"""

from __future__ import annotations

from itertools import groupby

from engineering_knowledge.chunking.errors import ChunkingError
from engineering_knowledge.chunking.markdown import split_into_sections
from engineering_knowledge.chunking.normalize import normalize_text
from engineering_knowledge.domain import Chunk, SectionPath


def chunk_plain_text(document_id: str, text: str, *, max_chunk_chars: int) -> tuple[Chunk, ...]:
    """Chunk plain text as a single root section.

    No heading parsing happens here: the entire normalized text is treated
    as ``SectionPath(())`` content, then split the same way a Markdown
    section is.
    """
    _validate_max_chunk_chars(max_chunk_chars)
    normalized = normalize_text(text)
    sections = [(SectionPath(), normalized)] if normalized else []
    return _build_chunks(document_id, sections, max_chunk_chars)


def chunk_markdown(document_id: str, text: str, *, max_chunk_chars: int) -> tuple[Chunk, ...]:
    """Chunk Markdown text using ATX heading structure.

    See ``chunking.markdown`` for exactly what heading syntax is
    recognized. Content before the first heading, or an entire document
    with no headings at all, belongs to ``SectionPath(())``.
    """
    _validate_max_chunk_chars(max_chunk_chars)
    normalized = normalize_text(text)
    sections = split_into_sections(normalized)
    return _build_chunks(document_id, sections, max_chunk_chars)


def _validate_max_chunk_chars(max_chunk_chars: int) -> None:
    if max_chunk_chars <= 0:
        raise ChunkingError("max_chunk_chars must be positive")


def _build_chunks(
    document_id: str,
    sections: list[tuple[SectionPath, str]],
    max_chunk_chars: int,
) -> tuple[Chunk, ...]:
    # section_occurrence counts appearances of the same canonical
    # section_path in document order. Keying by that path, not by a global
    # section counter, is what keeps an unrelated section elsewhere in the
    # document from perturbing this count.
    occurrence_counts: dict[str, int] = {}
    chunks: list[Chunk] = []
    ordinal = 0

    for section_path, section_text in sections:
        identity_key = section_path.as_identity_string()
        occurrence = occurrence_counts.get(identity_key, 0)
        occurrence_counts[identity_key] = occurrence + 1

        chunk_texts = _split_section(section_text, max_chunk_chars)
        for ordinal_in_section, chunk_text in enumerate(chunk_texts):
            chunks.append(
                Chunk.create(
                    document_id=document_id,
                    section_path=section_path,
                    section_occurrence=occurrence,
                    ordinal=ordinal,
                    ordinal_in_section=ordinal_in_section,
                    text=chunk_text,
                )
            )
            ordinal += 1

    return tuple(chunks)


def _split_section(section_text: str, max_chunk_chars: int) -> list[str]:
    lines = section_text.splitlines(keepends=True)
    units = _paragraph_units(lines)
    atoms = _atoms_for_units(units, max_chunk_chars)
    return _greedy_pack(atoms, max_chunk_chars)


def _paragraph_units(lines: list[str]) -> list[list[str]]:
    """Group lines into natural chunking units: a paragraph plus its trailing blank run.

    A run of blank lines with no preceding paragraph (leading blank lines
    in a section) becomes its own unit, since there is nothing to attach it
    to. This partitions ``lines`` without dropping or reordering any of
    them.
    """
    runs = [list(group) for _, group in groupby(lines, key=_is_blank_line)]
    units: list[list[str]] = []
    for run in runs:
        if _is_blank_line(run[0]) and units:
            units[-1].extend(run)
        else:
            units.append(list(run))
    return units


def _is_blank_line(line: str) -> bool:
    return line.rstrip("\n") == ""


def _atoms_for_units(units: list[list[str]], max_chunk_chars: int) -> list[str]:
    """Reduce each unit to a packable atom, falling back to finer boundaries only as needed."""
    atoms: list[str] = []
    for unit in units:
        unit_text = "".join(unit)
        if len(unit_text) <= max_chunk_chars:
            atoms.append(unit_text)
            continue

        for line in unit:
            if len(line) <= max_chunk_chars:
                atoms.append(line)
                continue
            atoms.extend(_hard_split(line, max_chunk_chars))

    return atoms


def _hard_split(text: str, max_chunk_chars: int) -> list[str]:
    return [text[i : i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]


def _greedy_pack(atoms: list[str], max_chunk_chars: int) -> list[str]:
    """Concatenate consecutive atoms into chunks no longer than the bound.

    Every atom handed in must already be no longer than ``max_chunk_chars``;
    ``_atoms_for_units`` guarantees that by construction.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for atom in atoms:
        if current and current_length + len(atom) > max_chunk_chars:
            chunks.append("".join(current))
            current = []
            current_length = 0
        current.append(atom)
        current_length += len(atom)

    if current:
        chunks.append("".join(current))

    return chunks
