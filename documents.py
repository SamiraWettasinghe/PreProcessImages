"""Pure logic for grouping a person's corrected crops into numbered documents.

The cropping app names each crop after its *source scan*, which never reveals which
*document* the crop belongs to. A document's pages can be scattered across non-adjacent
scans in arbitrary order, so membership cannot be expressed page-by-page in scan order.

This module backs a separate "document assignment" app: it lists every corrected crop
for one person, parses/encodes a document+page suffix on the filename, and renames the
files (keeping ``corrections.json`` in step) so the name reveals the document and page::

    002_Alber/S/Alber_026_S.png  ->  002_Alber/S/Alber_026_S_D04_p02.png
                                                          ^^^^ ^^^^ document 4, page 2

It is pure logic (no Tkinter) so it can be exercised headlessly, mirroring
``merge_sources.py``. Files are *moved*, never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from merge_sources import CORRECTED_DIRNAME, label_of, unique_path

# Same set the cropping app accepts (see app.py IMAGE_EXTS).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Trailing ``_D<doc>`` with an optional ``_p<page>``. Stripped before re-applying so a
# crop's stem never accumulates suffixes across repeated assignments.
SUFFIX_RE = re.compile(r"_D(\d+)(?:_p(\d+))?$")


def strip_doc_suffix(stem: str) -> str:
    """Return ``stem`` without any trailing ``_D##`` / ``_D##_p##`` document suffix."""
    return SUFFIX_RE.sub("", stem)


def parse_doc_suffix(stem: str) -> tuple[int | None, int | None]:
    """Return ``(doc, page)`` parsed from ``stem``'s suffix, or ``(None, None)``."""
    m = SUFFIX_RE.search(stem)
    if not m:
        return None, None
    doc = int(m.group(1))
    page = int(m.group(2)) if m.group(2) is not None else None
    return doc, page


def make_name(base_stem: str, doc: int, page: int, ext: str) -> str:
    """Build the canonical crop filename for ``doc``/``page`` (zero-padded to 2)."""
    return f"{base_stem}_D{doc:02d}_p{page:02d}{ext}"


@dataclass
class CropEntry:
    """One corrected crop belonging to a person, with any current doc assignment."""

    rel_out: str          # path relative to the corrected set, e.g. 002_Alber/S/Alber_026_S.png
    person: str           # top-level person folder, e.g. 002_Alber
    type_label: str       # type sub-folder ('' when directly under the person folder)
    base_stem: str        # filename stem with any doc suffix stripped
    ext: str              # file extension including the dot
    doc: int | None       # currently assigned document number (None = unassigned)
    page: int | None      # currently assigned page within the document


def list_people(out_dir: Path) -> list[str]:
    """Person folders under the corrected set that contain at least one image, sorted."""
    if not out_dir.is_dir():
        return []
    people = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if any(p.suffix.lower() in IMAGE_EXTS for p in child.rglob("*") if p.is_file()):
            people.append(child.name)
    return sorted(people)


def scan_person_crops(out_dir: Path, person: str) -> list[CropEntry]:
    """Every crop under ``out_dir/<person>/**``, with its current doc/page parsed.

    The ``person`` folder name doubles as the ``rel_src`` person for ``label_of`` (the
    type label is the crop's sub-folder under the person folder).
    """
    person_dir = out_dir / person
    if not person_dir.is_dir():
        return []
    rel_src = f"{person}/_"   # only the parent (the person folder) matters to label_of
    entries: list[CropEntry] = []
    for path in sorted(person_dir.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_out = path.relative_to(out_dir).as_posix()
        doc, page = parse_doc_suffix(path.stem)
        entries.append(CropEntry(
            rel_out=rel_out,
            person=person,
            type_label=label_of(rel_src, rel_out),
            base_stem=strip_doc_suffix(path.stem),
            ext=path.suffix,
            doc=doc,
            page=page,
        ))
    return entries


def plan_changes(
    entries: list[CropEntry],
    desired: dict[str, tuple[int | None, int | None, str]],
    out_dir: Path,
) -> list[tuple[str, str]]:
    """Compute ``(old_rel, new_rel)`` moves for the desired doc/page/type of each crop.

    ``desired`` maps ``rel_out`` -> ``(doc, raw_page, type_label)``:

    - ``doc`` is the document number, or ``None`` to leave the crop unassigned (its name
      then carries no ``_D##_p##`` suffix — re-clearing a doc strips an old suffix).
    - ``raw_page`` orders pages within a document; numbers are normalized to ``1..n``
      (ordered by raw page, then path) so gaps/ties collapse to a clean sequence.
    - ``type_label`` is the crop's classification sub-folder under the person folder
      (``''`` = directly in the person folder); changing it moves the crop.

    Both dimensions are handled uniformly: a crop moves if its folder *or* its name
    changes. No-ops are dropped; target collisions fall back to
    :func:`merge_sources.unique_path`.
    """
    by_rel = {e.rel_out: e for e in entries}

    # Normalize page numbers to 1..n within each document.
    docs: dict[int, list[str]] = {}
    for rel_out, (doc, _page, _typ) in desired.items():
        if rel_out in by_rel and doc is not None:
            docs.setdefault(doc, []).append(rel_out)
    page_of: dict[str, int] = {}
    for doc in docs:
        ordered = sorted(docs[doc], key=lambda r: (desired[r][1] if desired[r][1] is not None else 0, r))
        for page, rel_out in enumerate(ordered, start=1):
            page_of[rel_out] = page

    renames: list[tuple[str, str]] = []
    taken: set[str] = set()
    for rel_out in sorted(desired):
        if rel_out not in by_rel:
            continue
        e = by_rel[rel_out]
        doc, _raw, typ = desired[rel_out]
        typ = (typ or "").strip()
        parent = Path(e.person) / typ if typ else Path(e.person)
        if doc is not None:
            new_name = make_name(e.base_stem, doc, page_of[rel_out], e.ext)
        else:
            new_name = e.base_stem + e.ext
        new_rel = (parent / new_name).as_posix()
        if new_rel == rel_out:
            continue
        # Avoid clashing with an existing file or another crop's target this pass.
        if new_rel in taken or (out_dir / new_rel).exists():
            new_path = unique_path(out_dir / new_rel)
            new_rel = new_path.relative_to(out_dir).as_posix()
        taken.add(new_rel)
        renames.append((rel_out, new_rel))
    return renames


def apply_renames(
    out_dir: Path,
    corrections: dict[str, list[str]],
    renames: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Move each crop on disk and update ``corrections`` in place.

    Returns ``(corrections, unlinked)`` where ``unlinked`` lists any ``old_rel`` that had
    no entry in ``corrections.json`` (the file is still renamed; just unreferenced).
    Files are moved, never deleted. ``corrections`` maps original_rel -> [corrected_rel].
    """
    # Reverse index: corrected_rel -> (original_key, index in its list).
    reverse: dict[str, tuple[str, int]] = {}
    for orig_key, outs in corrections.items():
        for i, rel in enumerate(outs):
            reverse[rel] = (orig_key, i)

    unlinked: list[str] = []
    old_parents: set[Path] = set()
    for old_rel, new_rel in renames:
        src = out_dir / old_rel
        dst = out_dir / new_rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = unique_path(dst)
            new_rel = dst.relative_to(out_dir).as_posix()
        old_parents.add(src.parent)
        src.rename(dst)
        if old_rel in reverse:
            orig_key, idx = reverse[old_rel]
            corrections[orig_key][idx] = new_rel
            reverse[new_rel] = (orig_key, idx)
        else:
            unlinked.append(old_rel)

    # Tidy up any classification sub-folder left empty by a move (deepest first).
    for parent in sorted(old_parents, key=lambda p: len(p.parts), reverse=True):
        if parent != out_dir and parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass
    return corrections, unlinked
