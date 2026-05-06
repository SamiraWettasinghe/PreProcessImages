"""Tkinter GUI for picking a 4-corner region and rectifying it."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from transform import four_point_transform


POINT_RADIUS = 6
POINT_COLOR = "#ff3344"
POINT_OUTLINE = "#ffffff"
LINE_COLOR = "#22ff88"
LABEL_COLOR = "#ffeb3b"
HIT_RADIUS = 14      # canvas-px grab zone, larger than visual radius
CANVAS_BUFFER = 150  # dark padding (canvas px) around the image for out-of-bounds points

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_HERE           = Path(__file__).resolve().parent
BASE_DIR        = _HERE / "Akten_selektiert"
OUT_DIR         = _HERE / "Akten_selektiert_corrected"
LABELS_FILE     = _HERE / "labels.json"
CORRECTIONS_FILE = _HERE / "corrections.json"


class PerspectiveApp:
    """Main application window.

    Workflow:
        1. App loads all images from Akten_selektiert/ on startup.
        2. Click the four corners of the text region (any order); drag to adjust.
           The right pane shows the existing corrected version (if any).
        3. Click Apply — a label dialog appears. Choose / type a label (or leave
           blank for no category). The rectified image is saved under
           Akten_selektiert_corrected/<label>/<relative_path>.
           Re-applying deletes the old corrected file first.
        4. Use Prev / Next (or ← →) to move through the image list.
           Points are remembered per image so you can revisit and re-apply.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Perspective Correction Tool")

        # Left-pane (original) state
        self.image_path: str | None = None
        self.original: np.ndarray | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.scale: float = 1.0
        self.points: list[tuple[float, float]] = []
        self._drag_idx: int | None = None

        # Right-pane (corrected) state
        self.corrected: np.ndarray | None = None
        self.tk_corrected: ImageTk.PhotoImage | None = None

        # Navigation state
        self._image_list: list[Path] = []
        self._current_idx: int = 0
        self._saved_points: dict[str, list[tuple[float, float]]] = {}

        # Persistent data
        self._labels: list[str] = []       # dropdown history
        self._corrections: dict[str, str] = {}  # rel_src (posix) → rel_out (posix, within OUT_DIR)

        self._resize_job: str | None = None
        self._load_labels()
        self._load_corrections()
        self._build_ui()
        self.root.bind("<Configure>", self._on_window_resize)

        self._load_image_list()
        if self._image_list:
            self._go_to(0)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="◀ Prev", command=self.go_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next ▶", command=self.go_next).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(toolbar, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Apply", command=self.apply_correction).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._nav_label = ttk.Label(toolbar, text="— / —", width=14)
        self._nav_label.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(toolbar, text="Loading…")
        self.status.pack(side=tk.LEFT, padx=8)

        # ---- two-pane content area ----
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(content)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left_frame, text="Original", anchor="center",
                  font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(2, 0))
        self.canvas = tk.Canvas(
            left_frame, bg="#202225", cursor="crosshair", highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Separator(content, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        right_frame = ttk.Frame(content)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right_frame, text="Corrected", anchor="center",
                  font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(2, 0))
        self.canvas_right = tk.Canvas(
            right_frame, bg="#202225", cursor="arrow", highlightthickness=0,
        )
        self.canvas_right.pack(fill=tk.BOTH, expand=True)

        # Left canvas interactions
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)

        self.root.bind("<Control-r>", lambda _e: self.reset_points())
        self.root.bind("<Return>",    lambda _e: self.apply_correction())
        self.root.bind("<Left>",      lambda _e: self.go_prev())
        self.root.bind("<Right>",     lambda _e: self.go_next())

    # -------------------------------------------------- Persistent storage --

    def _load_labels(self) -> None:
        try:
            self._labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._labels = []

    def _save_labels(self) -> None:
        LABELS_FILE.write_text(
            json.dumps(self._labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_corrections(self) -> None:
        try:
            self._corrections = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._corrections = {}

    def _save_corrections(self) -> None:
        CORRECTIONS_FILE.write_text(
            json.dumps(self._corrections, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --------------------------------------------------------- Image list ---

    def _load_image_list(self) -> None:
        if not BASE_DIR.is_dir():
            messagebox.showerror("Not found", f"Source directory not found:\n{BASE_DIR}")
            return
        self._image_list = sorted(
            p for p in BASE_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        self._update_nav_label()

    def _corrected_path(self, src: Path) -> Path | None:
        """Return the on-disk path of src's corrected version, or None if it doesn't exist."""
        rel_src = src.relative_to(BASE_DIR).as_posix()
        rel_out = self._corrections.get(rel_src)
        if rel_out is not None:
            p = OUT_DIR / Path(rel_out)
            return p if p.exists() else None
        # Backward-compat: check the unlabeled location used before the label feature.
        p = OUT_DIR / src.relative_to(BASE_DIR)
        return p if p.exists() else None

    def _is_corrected(self, src: Path) -> bool:
        return self._corrected_path(src) is not None

    # --------------------------------------------------------- Navigation ---

    def _go_to(self, idx: int) -> None:
        if not self._image_list:
            return

        if self.image_path is not None:
            self._saved_points[self.image_path] = list(self.points)

        self._current_idx = max(0, min(idx, len(self._image_list) - 1))
        src = self._image_list[self._current_idx]

        try:
            data = np.fromfile(str(src), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:
            self._set_status(f"Could not read: {exc}")
            return
        if image is None:
            self._set_status(f"Could not decode {src.name}.")
            return

        self.image_path = str(src)
        self.original = image
        self.points = list(self._saved_points.get(str(src), []))
        self._drag_idx = None

        # Load corrected version from disk if it exists.
        cp = self._corrected_path(src)
        if cp is not None:
            try:
                cdata = np.fromfile(str(cp), dtype=np.uint8)
                self.corrected = cv2.imdecode(cdata, cv2.IMREAD_COLOR)
            except Exception:
                self.corrected = None
        else:
            self.corrected = None

        self._render_both()
        self._update_nav_label()
        self._set_load_status(src)

    def _set_load_status(self, src: Path) -> None:
        cp = self._corrected_path(src)
        if cp is not None and self.points:
            self._set_status(f"{src.name}  [corrected: {cp.parent.name}] — adjust & re-apply, or press Next ▶")
        elif cp is not None:
            self._set_status(f"{src.name}  [corrected: {cp.parent.name}] — click corners to re-correct, or press Next ▶")
        elif self.points:
            self._set_status(f"{src.name} — points restored, drag to adjust or click Apply.")
        else:
            self._set_status(f"{src.name} — click 4 corners of the text region.")

    def _update_nav_label(self) -> None:
        total = len(self._image_list)
        if total == 0:
            self._nav_label.config(text="— / —")
            return
        done = "✓" if self._is_corrected(self._image_list[self._current_idx]) else " "
        self._nav_label.config(text=f"{done} {self._current_idx + 1} / {total}")

    def go_prev(self) -> None:
        self._go_to(self._current_idx - 1)

    def go_next(self) -> None:
        self._go_to(self._current_idx + 1)

    # --------------------------------------------------------------- Render --

    def _render_both(self) -> None:
        self._render_image()
        self._render_corrected()

    def _render_image(self) -> None:
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        usable_w = max(cw - 2 * CANVAS_BUFFER, 1)
        usable_h = max(ch - 2 * CANVAS_BUFFER, 1)
        self.scale = min(usable_w / w, usable_h / h, 1.0)
        new_w = max(int(w * self.scale), 1)
        new_h = max(int(h * self.scale), 1)

        rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_BUFFER, CANVAS_BUFFER, anchor=tk.NW, image=self.tk_image)
        self._redraw_points()

    def _render_corrected(self) -> None:
        self.canvas_right.delete("all")
        cw = max(self.canvas_right.winfo_width(), 100)
        ch = max(self.canvas_right.winfo_height(), 100)

        if self.corrected is None:
            self.canvas_right.create_text(
                cw // 2, ch // 2,
                text="No correction yet",
                fill="#666677", font=("Arial", 14),
            )
            return

        h, w = self.corrected.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        rgb = cv2.cvtColor(self.corrected, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_corrected = ImageTk.PhotoImage(pil)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.tk_corrected)
        self.canvas_right.create_text(
            4, ch - 4, anchor=tk.SW,
            text=f"{w} × {h} px",
            fill="#aaaaaa", font=("Arial", 9),
        )

    def _on_window_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._render_both)

    # ------------------------------------------------------- Point handling --

    def _hit_test(self, cx: float, cy: float) -> int | None:
        best_idx, best_dist = None, float(HIT_RADIUS)
        for i, (ox, oy) in enumerate(self.points):
            dx = ox * self.scale + CANVAS_BUFFER - cx
            dy = oy * self.scale + CANVAS_BUFFER - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _on_click(self, event: tk.Event) -> None:
        if self.original is None:
            return
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self._drag_idx = hit
            return
        if len(self.points) >= 4:
            self._set_status("Drag a point to reposition it, or Reset Points to start over.")
            return
        self.points.append(((event.x - CANVAS_BUFFER) / self.scale,
                             (event.y - CANVAS_BUFFER) / self.scale))
        self._redraw_points()
        if len(self.points) == 4:
            self._set_status("4 points set — drag to adjust, or click Apply.")
        else:
            self._set_status(f"Selected {len(self.points)}/4 points.")

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_idx is None or self.original is None:
            return
        ox = (event.x - CANVAS_BUFFER) / self.scale
        oy = (event.y - CANVAS_BUFFER) / self.scale
        self.points[self._drag_idx] = (ox, oy)
        self._redraw_points()

    def _on_release(self, _event: tk.Event) -> None:
        if self._drag_idx is not None:
            self._drag_idx = None
            if len(self.points) == 4:
                self._set_status("4 points set — drag to adjust, or click Apply.")

    def _on_hover(self, event: tk.Event) -> None:
        cursor = "fleur" if self._hit_test(event.x, event.y) is not None else "crosshair"
        self.canvas.config(cursor=cursor)

    def reset_points(self) -> None:
        self.points.clear()
        self._redraw_points()
        if self.original is not None:
            self._set_status("Click the 4 corners of the text region (any order).")

    def _redraw_points(self) -> None:
        self.canvas.delete("overlay")
        if not self.points:
            return

        def to_canvas(ox, oy):
            return ox * self.scale + CANVAS_BUFFER, oy * self.scale + CANVAS_BUFFER

        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                x1, y1 = to_canvas(*self.points[i])
                x2, y2 = to_canvas(*self.points[i + 1])
                self.canvas.create_line(x1, y1, x2, y2, fill=LINE_COLOR, width=2, tags="overlay")
        if len(self.points) == 4:
            x1, y1 = to_canvas(*self.points[-1])
            x2, y2 = to_canvas(*self.points[0])
            self.canvas.create_line(x1, y1, x2, y2, fill=LINE_COLOR, width=2, tags="overlay")
        for i, pt in enumerate(self.points, start=1):
            cx, cy = to_canvas(*pt)
            self.canvas.create_oval(
                cx - POINT_RADIUS, cy - POINT_RADIUS,
                cx + POINT_RADIUS, cy + POINT_RADIUS,
                fill=POINT_COLOR, outline=POINT_OUTLINE, width=2, tags="overlay",
            )
            self.canvas.create_text(
                cx + 12, cy - 12, text=str(i),
                fill=LABEL_COLOR, font=("Arial", 12, "bold"), tags="overlay",
            )

    # -------------------------------------------------------------- Apply ---

    def _filenames_for_folder(self, src: Path) -> list[str]:
        """Return sorted output filenames already used for images in src's source folder."""
        parent = src.relative_to(BASE_DIR).parent.as_posix()
        names: set[str] = set()
        for rel_src_key, rel_out_val in self._corrections.items():
            if Path(rel_src_key).parent.as_posix() == parent:
                names.add(Path(rel_out_val).name)
        return sorted(names)

    def _ask_save_options(self, src: Path) -> tuple[str, str] | None:
        """Modal dialog for label + filename.

        Returns (label, filename) — label may be empty (no subdirectory) — or
        None if the user cancelled (aborts the save).
        Tab / Enter move between fields; Enter in filename submits.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Options")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Label  (leave blank for no category):").pack(
            padx=16, pady=(14, 2), anchor="w"
        )
        label_combo = ttk.Combobox(dialog, values=self._labels, width=36)
        label_combo.pack(padx=16, pady=(0, 10))
        label_combo.focus_set()

        ttk.Label(dialog, text="Filename:").pack(padx=16, pady=(0, 2), anchor="w")
        name_combo = ttk.Combobox(dialog, values=self._filenames_for_folder(src), width=36)
        name_combo.set(src.name)
        name_combo.pack(padx=16, pady=(0, 10))

        result: list[tuple[str, str] | None] = [None]

        def on_ok(_event=None) -> None:
            label    = label_combo.get().strip()
            filename = name_combo.get().strip() or src.name
            # Preserve original extension if the user omitted it.
            if not Path(filename).suffix:
                filename += src.suffix
            result[0] = (label, filename)
            dialog.destroy()

        def on_cancel(_event=None) -> None:
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=(0, 14))
        ttk.Button(btn_frame, text="OK",     command=on_ok,     width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=6)

        # Enter in label field moves to filename; Enter in filename submits.
        label_combo.bind("<Return>", lambda _e: name_combo.focus_set())
        name_combo.bind("<Return>", on_ok)
        dialog.bind("<Escape>", on_cancel)

        # Centre over parent window.
        self.root.update_idletasks()
        dialog.update_idletasks()
        px = self.root.winfo_x() + (self.root.winfo_width()  - dialog.winfo_width())  // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{px}+{py}")

        dialog.wait_window()
        return result[0]

    def apply_correction(self) -> None:
        if self.original is None:
            messagebox.showinfo("No image", "No image loaded.")
            return
        if len(self.points) != 4:
            messagebox.showinfo(
                "Need 4 points",
                f"Please click 4 corners (you have {len(self.points)}).",
            )
            return

        pts = np.array(self.points, dtype="float32")
        try:
            warped = four_point_transform(self.original, pts)
        except Exception as exc:
            messagebox.showerror("Transform failed", str(exc))
            return

        src = Path(self.image_path)
        options = self._ask_save_options(src)
        if options is None:
            return  # user cancelled — abort
        label, filename = options

        rel_src = src.relative_to(BASE_DIR).as_posix()

        # Delete the old corrected file (if any) before saving the new one.
        old_rel_out = self._corrections.get(rel_src)
        if old_rel_out is not None:
            old_file = OUT_DIR / Path(old_rel_out)
            if old_file.exists():
                old_file.unlink()
        else:
            # Backward-compat: remove unlabeled file from before the label feature.
            old_unlabeled = OUT_DIR / src.relative_to(BASE_DIR)
            if old_unlabeled.exists():
                old_unlabeled.unlink()

        # Compute new output path: OUT_DIR/<person_dir>/<label>/<filename>
        rel_src_path = src.relative_to(BASE_DIR)
        if label:
            rel_out = (rel_src_path.parent / label / filename).as_posix()
        else:
            rel_out = (rel_src_path.parent / filename).as_posix()
        out = OUT_DIR / Path(rel_out)
        out.parent.mkdir(parents=True, exist_ok=True)

        ext = out.suffix.lower() or ".png"
        ok, buf = cv2.imencode(ext, warped)
        if not ok:
            messagebox.showerror("Save failed", "Could not encode image.")
            return
        try:
            buf.tofile(str(out))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        # Persist label and correction mapping.
        if label and label not in self._labels:
            self._labels.append(label)
            self._labels.sort()
            self._save_labels()
        self._corrections[rel_src] = rel_out
        self._save_corrections()

        self.corrected = warped
        self._render_corrected()
        self._update_nav_label()
        label_display = f"[{label}]" if label else "[no label]"
        self._set_status(f"Saved {label_display} → {out.relative_to(_HERE)}")

    # ------------------------------------------------------------- Helpers --

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
