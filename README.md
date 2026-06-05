# Perspective Correction Tool

A Python/Tkinter GUI for batch-cropping regions from photographed or scanned
documents and **rectifying** them — removing perspective distortion so the text
appears perpendicular to the camera (the top of the page no longer looks
smaller than the bottom). It is built around a folder of archival document
images and remembers your work between sessions.

It supports multiple regions per page, an optional OCR-oriented preprocessing
stage, zoom/pan, and a bilingual interface (**English / German**).

## Why 4 points instead of a bounding box?

A standard axis-aligned bounding box can only **crop**. To undo perspective you
need a *quadrilateral* — four corners that don't have to form a rectangle in the
input image. The tool maps that quadrilateral onto a true rectangle using
`cv2.getPerspectiveTransform` + `cv2.warpPerspective`, which is the correct way
to flatten a tilted page.

If you really only want a rectangular crop, you can still do that — just put your
4 points at the corners of an upright rectangle. But for documents photographed
at an angle, picking the actual visual corners of the text block is what gives
you a proper "scanned-looking" output.

## Install

```bash
pip install -r requirements.txt
```

Dependencies: OpenCV, NumPy, Pillow. Tkinter ships with Python on Windows and
macOS. On Linux you may need `sudo apt install python3-tk` (Debian/Ubuntu) or the
equivalent for your distro.

## Run

```bash
python main.py
```

On startup a small dialog asks you to pick the interface language
(**English** / **Deutsch**) before the main window opens. The tool then scans
the input folder and loads the first image automatically — there is no "open
file" step.

## Folders

The tool works against two sibling folders next to the scripts:

| Folder                        | Role                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| `Akten_selektiert/`           | **Input.** Scanned recursively for images.                  |
| `Akten_selektiert_corrected/` | **Output.** Rectified images, mirroring the input subtree.  |

Recognised image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`,
`.webp`.

## Language

The whole interface — toolbar, preprocessing panel and its tooltips, dialogs,
and status messages — is available in **English** and **German**.

The language is chosen once, at startup, and stays fixed for that session. There
is no in-app switch; to change language, **restart the application** and pick the
other option. Closing the picker without choosing defaults to English.

## Usage

The window shows the **original** image on the left and the **corrected**
result(s) on the right.

1. **Navigate** with `◀ Prev` / `Next ▶` (or the `←` / `→` arrow keys). Type a
   number in the page box and press `Enter` to jump. A `✓` marks images that
   already have a saved correction.
2. **Draw a box** by clicking the 4 corners of the region you want. Order
   doesn't matter — the corners are sorted into TL/TR/BR/BL automatically. Each
   corner gets a numbered dot and the quadrilateral is outlined. Drag a dot to
   nudge it.
3. **Add more boxes** on the same page by clicking another 4 corners. Each box
   gets its own colour and is exported as a separate output image.
4. **Apply** (button or `Enter`). A *Save Options* dialog appears: for each box
   you can set an optional **Label** (creates a sub-folder in the output tree)
   and a **Filename**. The rectified crops are written to
   `Akten_selektiert_corrected/` and shown in the right pane.
5. **Delete a correction** with `Delete Correction`. Tick which saved output(s)
   to remove, confirm, and the tool deletes the image file(s), drops the entry
   from `corrections.json`, and removes the matching box.

Other controls:

- **Right-click a corner** for a menu to delete that single point or the whole
  box.
- **Reset Points** clears all boxes on the current image.
- **Zoom** with `−` / `Fit` / `+`, the mouse wheel (zooms toward the cursor), or
  the keyboard. **Pan** by dragging with the middle mouse button.

### Keyboard shortcuts

| Key            | Action                          |
| -------------- | ------------------------------- |
| `←` / `→`      | Previous / next image           |
| `Enter`        | Apply correction                |
| `Ctrl+R`       | Reset points                    |
| `+` / `=`      | Zoom in                         |
| `−`            | Zoom out                        |
| `0`            | Fit to window                   |

## Preprocessing (optional)

Tick **Preprocess** to open a panel of cleanup steps aimed at making output more
OCR-friendly. The right pane updates live as you adjust the sliders, and every
control has a hover tooltip explaining what it does and when to use it.

- **Denoise** — removes scanner grain (`h` = strength).
- **CLAHE** — evens out uneven lighting (`Clip`, `Grid`).
- **Threshold** — convert to black-and-white: `None`, `Otsu`, or `Adaptive`
  (with `Block` size and `C` bias).
- **Morphology** — closes small gaps in letter strokes (`Kernel`).
- **Upscale** — enlarges the image before OCR (`Factor`).

`Apply` saves the preprocessed result. `Save Preproc` re-saves the current
preprocessing to the existing output files without re-warping. The parameter
values are remembered in `preprocess_settings.json`.

## Persistence

Your work is saved alongside the scripts so it survives a restart:

| File                       | Contents                                                       |
| -------------------------- | -------------------------------------------------------------- |
| `corrections.json`         | Maps each source image to its corrected output file(s).        |
| `boxes.json`               | The corner points you placed for each image.                   |
| `labels.json`              | Labels you've used (offered as suggestions in Save Options).   |
| `preprocess_settings.json` | The current preprocessing parameter values.                    |

These are local artifacts and are git-ignored.

## Project layout

```
Masters/
├── main.py          # entry point — `python main.py`
├── app.py           # Tkinter GUI (PerspectiveApp class) + translations
├── transform.py     # pure perspective-warp logic, no GUI
├── requirements.txt
└── README.md
```

`transform.py` is intentionally GUI-free, so you can also call
`four_point_transform(image, pts)` from a script or notebook.

## Notes

- Corner ordering is angle-based (each point's angle around the centroid), which
  stays stable even at extreme angles like 45° — where the classic `x+y / y−x`
  heuristic can collapse two corners and produce a blank warp.
- Output dimensions are chosen as the *larger* of each pair of opposite sides,
  so the rectified image preserves detail rather than downsampling.
- File I/O uses `np.fromfile` / `cv2.imencode` + `tofile` so non-ASCII paths
  (Umlauts, etc.) work on Windows too — useful for the *Meldebogen*-style German
  archival documents this was built around.
- Cubic interpolation (`cv2.INTER_CUBIC`) is used for the warp, which generally
  gives the cleanest text.
