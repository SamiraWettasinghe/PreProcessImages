# CLAUDE.md

Guidance for Claude Code working in this repository.

> Note: the `CLAUDE.md` one level up (`jana/CLAUDE.md`) describes a *different*,
> unrelated project (sentiment analysis). It does not apply here. This folder is the
> **Perspective Correction Tool**.

## What this is

A Python/Tkinter desktop app for batch-cropping regions from scanned archival
documents and **rectifying** them (perspective-warping a 4-corner quadrilateral onto a
true rectangle via OpenCV). Bilingual UI (English / German), zoom/pan, optional
OCR-oriented preprocessing, and per-image persistence. See `README.md` for the full
user-facing manual.

## Run

```bash
python main.py            # dev run
./launch.sh               # what the desktop icon runs (uses ./venv, logs to last_run.log)
pip install -r requirements.txt   # deps: OpenCV, NumPy, Pillow (+ system Tk)
```

There is a `venv/` in this folder used by `launch.sh`. A startup dialog picks the UI
language (fixed for the session — restart to change).

No automated test suite. Verify changes by exercising the real code paths headlessly
against a throwaway sandbox (build a temp `Akten_selektiert*` tree, point the module
constants at it, stub the modal dialogs) rather than clicking through the GUI.

## Code map

| File               | Role |
| ------------------ | ---- |
| `main.py`          | Entry point; creates the Tk root and `PerspectiveApp`. |
| `app.py`           | The whole GUI: one `PerspectiveApp` class + the `TRANSLATIONS` dict. |
| `transform.py`     | Pure `four_point_transform(image, pts)` — no GUI; callable standalone. |
| `merge_sources.py` | Pure logic (no Tk) for discovering/merging external `_X` correction folders. |

`app.py` is a single large class. Everything — rendering, navigation, point editing,
preprocessing, save/delete, and the external-merge UI — lives there. Keep new
non-GUI logic in a separate module (as `merge_sources.py` does) so it stays testable.

## Data model

The app pages over originals in `Akten_selektiert/` (left pane) and writes rectified
crops to `Akten_selektiert_corrected/` (right pane). Three JSON files persist work and
are **git-ignored** local artifacts:

- `corrections.json` — `original_rel → [corrected_rel, ...]` (relative paths; portable).
  The source of truth for "what has been corrected". A box may be sorted into a
  **sub-folder/label** (e.g. `001_Aichele/M/Aichele_006.png`); the label is taken from
  the **output path**, never parsed from the filename.
- `boxes.json` — `original_abs → [[x,y] × 4 per box]`. Keyed by absolute path.
- `labels.json` — sub-folder labels offered as suggestions in the Save Options dialog.
- `preprocess_settings.json` — current preprocessing slider values.

## External `_X` merge feature

Corrections made on other laptops are copied into sibling folders named
`Akten_selektiert_corrected_<tag>` (e.g. `..._SynLaptop`), each a self-contained copy
with its own nested `Akten_selektiert_corrected/` + `corrections.json` / `boxes.json` /
`labels.json`. The tool merges these into the main set page-by-page:

- `discover_external_sources(here)` finds every `Akten_selektiert_corrected_*` sibling,
  **explicitly excluding** the archive folder (`ARCHIVE_DIRNAME`). The archive name also
  matches the `_*` prefix, so the exclusion is by name, not by the double underscore.
- `normalize_original_key()` reduces any original path — including the foreign Windows
  paths (`C:\...\Akten_selektiert\...`) in an `_X`'s `boxes.json` — to a portable key by
  splitting on the `Akten_selektiert/` marker.
- In the GUI, **Show Possible Duplicates** opens a comparison: the original with each
  source's boxes overlaid (main solid, external dashed; a box only draws where that
  source actually saved corner points), and a per-crop card list (source · sub-folder ·
  filename + checkbox, all unticked by default).
- Kept crops re-prompt for sub-folder/label (reusing the Apply dialog) and are written
  into main; un-kept crops and the `_X` copies of kept crops are **moved** to
  `Akten_selektiert_corrected__archive/<source>/...` — never deleted — and removed from
  the owning source's `corrections.json`, so the button self-clears on revisit.
- **Unreachable…** reports `_X` corrections whose original is absent locally (cannot be
  reached page-by-page).

## Conventions

- **i18n**: every user-facing string goes through `self.t(key, **kw)` (or `self.tl` for
  lists) and must have an entry in **both** the `en` and `de` blocks of `TRANSLATIONS`.
- **Non-ASCII paths**: read/write images with `np.fromfile` + `cv2.imdecode` /
  `cv2.imencode` + `tofile` (the documents have Umlauts; this keeps Windows happy).
- **Never delete user data**: archive (move) rather than `unlink`, except where the
  existing `Delete Correction` flow already deletes by explicit user confirmation.
- **Corner ordering** is angle-based (stable at extreme angles); output size is the
  larger of each opposite side; warp uses `cv2.INTER_CUBIC`.
