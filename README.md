# Perspective Correction Tool

A small Python GUI for cropping a region from a photographed document and
**rectifying** it — i.e. removing perspective distortion so the text appears
perpendicular to the camera (top of the page no longer looks smaller than
the bottom).

## Why 4 points instead of a bounding box?

A standard axis-aligned bounding box can only **crop**. To undo perspective
you need a *quadrilateral* — four corners that don't have to form a rectangle
in the input image. The tool maps that quadrilateral onto a true rectangle
using `cv2.getPerspectiveTransform` + `cv2.warpPerspective`, which is the
correct way to flatten a tilted page.

If you really only want a rectangular crop, you can still do that — just put
your 4 points at the corners of an upright rectangle. But for documents
photographed at an angle, picking the actual visual corners of the text
block is what gives you a proper "scanned-looking" output.

## Install

```bash
pip install -r requirements.txt
```

Tkinter ships with Python on Windows and macOS. On Linux you may need
`sudo apt install python3-tk` (Debian/Ubuntu) or the equivalent for your
distro.

## Run

```bash
python main.py
```

## Usage

1. **Open Image…** — load your photo (JPG, PNG, TIFF, etc.).
2. **Click 4 corners** of the text region you care about. Order doesn't
   matter — the app sorts them into TL / TR / BR / BL automatically. A
   numbered red dot appears at each click and the polygon is drawn in
   green.
3. **Apply & Save…** — pick a destination. The rectified image opens in a
   preview window and is written to disk.
4. **Reset Points** clears the selection if you misclick.

Keyboard shortcuts: `Ctrl+O` open · `Ctrl+R` reset · `Enter` apply.

## Project layout

```
perspective_corrector/
├── main.py          # entry point — `python main.py`
├── app.py           # Tkinter GUI (PerspectiveApp class)
├── transform.py     # pure perspective-warp logic, no GUI
├── requirements.txt
└── README.md
```

`transform.py` is intentionally GUI-free, so you can also call
`four_point_transform(image, pts)` from a script or notebook.

## Notes

- Output dimensions are chosen as the *larger* of each pair of opposite
  sides, so the rectified image preserves detail rather than downsampling.
- File I/O uses `np.fromfile` / `cv2.imencode` + `tofile` so non-ASCII
  paths (Umlauts, etc.) work on Windows too — useful for the
  *Meldebogen*-style German archival documents this was built around.
- Cubic interpolation (`cv2.INTER_CUBIC`) is used for the warp, which
  generally gives the cleanest text.
