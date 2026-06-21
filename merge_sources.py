"""Discovery and merging of corrections from external ``_X`` folders.

The canonical corrected set lives in ``Akten_selektiert_corrected/``. Work done on
other laptops is copied into sibling ``Akten_selektiert_corrected_<tag>/`` folders,
each a self-contained copy: a nested ``Akten_selektiert_corrected/`` with the crops,
plus its own ``corrections.json`` / ``boxes.json`` / ``labels.json``.

This module locates those external sources, lists the crops each one holds for a
given original image, and supports moving crops into a non-destructive archive while
keeping each source's ``corrections.json`` in step. It is pure logic (no Tkinter) so
it can be tested in isolation and keeps ``app.py`` lean.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil


BASE_DIRNAME      = "Akten_selektiert"
CORRECTED_DIRNAME = "Akten_selektiert_corrected"
# The archive name also matches the `Akten_selektiert_corrected_*` source prefix, so it
# is excluded explicitly by name in discover_external_sources() (not by the double
# underscore, which only makes it sort/read distinctly).
ARCHIVE_DIRNAME   = "Akten_selektiert_corrected__archive"


def normalize_original_key(path_str: str) -> str:
    """Reduce any original-image path to its repo-relative key.

    Handles local POSIX paths and foreign Windows paths alike, e.g.::

        C:\\Users\\janal\\...\\Akten_selektiert\\001_Aichele\\Aichele_006.png
        /home/sam/.../Akten_selektiert/001_Aichele/Aichele_006.png
        001_Aichele/Aichele_006.png

    all collapse to ``001_Aichele/Aichele_006.png``. The marker is matched with a
    trailing slash so ``Akten_selektiert_corrected/`` never matches by accident.
    """
    s = str(path_str).replace("\\", "/")
    marker = BASE_DIRNAME + "/"
    idx = s.find(marker)
    if idx != -1:
        return s[idx + len(marker):]
    return s.lstrip("/")


def group_box(flat_pts: list | None, i: int) -> list | None:
    """Return the i-th 4-point box from a flat point list, or None if unavailable."""
    if not flat_pts:
        return None
    grp = flat_pts[i * 4:i * 4 + 4]
    return grp if len(grp) == 4 else None


def label_of(rel_src: str, rel_out: str) -> str:
    """Sub-folder/label a crop was sorted into ('' if directly in the person folder).

    ``rel_src`` is the original's relative key (``001_Aichele/Aichele_006.png``);
    ``rel_out`` is the crop's relative path (``001_Aichele/SD/Aichele_006.png``).
    """
    person   = Path(rel_src).parent.as_posix()
    out_dir  = Path(rel_out).parent.as_posix()
    if out_dir in (person, ".", ""):
        return ""
    if out_dir.startswith(person + "/"):
        return out_dir[len(person) + 1:]
    return Path(rel_out).parent.name


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@dataclass
class CropInfo:
    """One corrected crop of an original, from a particular source."""
    source_tag: str          # "main" or an external folder's suffix, e.g. "SynLaptop"
    abs_file: Path           # where the crop currently lives on disk
    rel_out: str             # crop path relative to its source's corrected dir
    label: str               # sub-folder/label it was sorted into ("" = none)
    box_pts: list | None     # the 4 corner points that produced it, or None
    exists: bool             # whether abs_file is present on disk


@dataclass
class Source:
    """An external ``_X`` corrected folder and its metadata."""
    tag: str
    root: Path                       # the `_X` folder itself
    corrected_dir: Path              # <root>/Akten_selektiert_corrected
    corrections: dict                # original_rel -> [crop_rel, ...]
    boxes_by_rel: dict               # original_rel -> flat [[x,y], ...]
    corrections_path: Path

    def has(self, rel_src: str) -> bool:
        return bool(self.corrections.get(rel_src))

    def crops_for(self, rel_src: str) -> list[CropInfo]:
        rel_outs = self.corrections.get(rel_src) or []
        pts = self.boxes_by_rel.get(rel_src)
        out: list[CropInfo] = []
        for i, rel_out in enumerate(rel_outs):
            abs_file = self.corrected_dir / rel_out
            out.append(CropInfo(
                source_tag=self.tag,
                abs_file=abs_file,
                rel_out=rel_out,
                label=label_of(rel_src, rel_out),
                box_pts=group_box(pts, i),
                exists=abs_file.exists(),
            ))
        return out

    def remove_entry(self, rel_src: str, rel_out: str) -> None:
        """Drop one crop from this source; remove the original entirely if it was the last."""
        lst = self.corrections.get(rel_src)
        if not lst:
            return
        if rel_out in lst:
            lst.remove(rel_out)
        if lst:
            self.corrections[rel_src] = lst
        else:
            self.corrections.pop(rel_src, None)
            self.boxes_by_rel.pop(rel_src, None)

    def save(self) -> None:
        """Persist this source's corrections.json (boxes.json is left untouched on disk:
        it is only read for overlays, and stale entries are harmless once the matching
        corrections entry is gone)."""
        try:
            self.corrections_path.write_text(
                json.dumps(self.corrections, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def discover_external_sources(here: Path) -> list[Source]:
    """Find every ``Akten_selektiert_corrected_<tag>`` sibling (excluding the archive)."""
    sources: list[Source] = []
    prefix = CORRECTED_DIRNAME + "_"
    for child in sorted(here.iterdir()):
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        if child.name == ARCHIVE_DIRNAME:
            continue
        tag = child.name[len(prefix):]

        corrected_dir = child / CORRECTED_DIRNAME
        if not corrected_dir.is_dir():
            corrected_dir = child  # tolerate crops sitting directly in the _X folder

        corr_path = child / "corrections.json"
        raw_corr  = _load_json(corr_path, {})
        corrections = {
            k: ([v] if isinstance(v, str) else list(v))
            for k, v in raw_corr.items()
        }
        raw_boxes = _load_json(child / "boxes.json", {})
        boxes_by_rel = {normalize_original_key(k): v for k, v in raw_boxes.items()}

        sources.append(Source(
            tag=tag, root=child, corrected_dir=corrected_dir,
            corrections=corrections, boxes_by_rel=boxes_by_rel,
            corrections_path=corr_path,
        ))
    return sources


def orphan_corrections(
    sources: list[Source], local_keys: set[str]
) -> list[tuple[str, str, list[str]]]:
    """External corrections whose original image is absent locally (unreachable page-by-page)."""
    out: list[tuple[str, str, list[str]]] = []
    for s in sources:
        for rel_src, rel_outs in sorted(s.corrections.items()):
            if rel_src not in local_keys:
                out.append((s.tag, rel_src, list(rel_outs)))
    return out


def unique_path(p: Path) -> Path:
    """Return ``p`` or, if it exists, ``p`` with a ``_dupN`` suffix that does not."""
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 2
    while True:
        cand = p.with_name(f"{stem}_dup{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def archive_move(abs_file: Path, archive_root: Path, tag: str, rel_out: str) -> Path | None:
    """Move ``abs_file`` into ``<archive_root>/<tag>/<rel_out>`` (never delete). Returns dest."""
    if not abs_file.exists():
        return None
    dest = unique_path(archive_root / tag / rel_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(abs_file), str(dest))
    return dest
