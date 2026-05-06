"""Tkinter GUI for picking 4-corner regions and rectifying them."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from transform import four_point_transform


POINT_RADIUS  = 6
POINT_COLOR   = "#ff3344"
POINT_OUTLINE = "#ffffff"
LABEL_COLOR   = "#ffeb3b"
HIT_RADIUS    = 14
CANVAS_BUFFER = 150
ZOOM_STEP     = 1.25   # factor per scroll tick
ZOOM_MIN      = 0.25
ZOOM_MAX      = 20.0

# One colour per box (cycles if more than 6 boxes).
BOX_COLORS = ["#22ff88", "#ff9922", "#22aaff", "#ff44cc", "#ffff44", "#aa44ff"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_HERE            = Path(__file__).resolve().parent
BASE_DIR         = _HERE / "Akten_selektiert"
OUT_DIR          = _HERE / "Akten_selektiert_corrected"
LABELS_FILE      = _HERE / "labels.json"
CORRECTIONS_FILE = _HERE / "corrections.json"


class PerspectiveApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Perspective Correction Tool")

        # Left-pane image state
        self.image_path: str | None = None
        self.original: np.ndarray | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        # Coordinate transform state
        # Effective scale = _base_scale * _zoom
        # Image top-left in canvas: (CANVAS_BUFFER + pan_x, CANVAS_BUFFER + pan_y)
        self._base_scale: float = 1.0   # computed to fit image in canvas
        self._zoom: float = 1.0         # user-controlled multiplier
        self.scale: float = 1.0         # = _base_scale * _zoom  (used everywhere)
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Pan gesture state
        self._panning: bool = False
        self._pan_start: tuple[int, int] = (0, 0)
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)

        # Points / boxes
        self.points: list[tuple[float, float]] = []
        self._drag_idx: int | None = None

        # Right-pane state
        self.corrected: list[np.ndarray] = []
        self.tk_corrected: ImageTk.PhotoImage | None = None

        # Navigation
        self._image_list: list[Path] = []
        self._current_idx: int = 0
        self._saved_points: dict[str, list[tuple[float, float]]] = {}

        # Persistent data
        self._labels: list[str] = []
        self._corrections: dict[str, list[str]] = {}

        self._resize_job: str | None = None
        self._zoom_job:   str | None = None
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

        ttk.Button(toolbar, text="◀ Prev",       command=self.go_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next ▶",       command=self.go_next).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Apply",        command=self.apply_correction).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="−",   width=2, command=self.zoom_out).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="Fit", width=3, command=self.zoom_fit).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="+",   width=2, command=self.zoom_in).pack(side=tk.LEFT, padx=1)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._nav_label = ttk.Label(toolbar, text="— / —", width=14)
        self._nav_label.pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(toolbar, text="Loading…")
        self.status.pack(side=tk.LEFT, padx=8)

        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(content)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left_frame, text="Original", anchor="center",
                  font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(2, 0))
        self.canvas = tk.Canvas(left_frame, bg="#202225", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Separator(content, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        right_frame = ttk.Frame(content)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right_frame, text="Corrected", anchor="center",
                  font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(2, 0))
        self.canvas_right = tk.Canvas(right_frame, bg="#202225", cursor="arrow",
                                      highlightthickness=0)
        self.canvas_right.pack(fill=tk.BOTH, expand=True)

        # Left-canvas bindings
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)

        # Pan: middle-click drag
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # Zoom: mouse wheel (cross-platform)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)   # Windows / macOS
        self.canvas.bind("<Button-4>",   self._on_mousewheel)   # Linux scroll up
        self.canvas.bind("<Button-5>",   self._on_mousewheel)   # Linux scroll down

        self.root.bind("<Control-r>", lambda _e: self.reset_points())
        self.root.bind("<Return>",    lambda _e: self.apply_correction())
        self.root.bind("<Left>",      lambda _e: self.go_prev())
        self.root.bind("<Right>",     lambda _e: self.go_next())
        self.root.bind("<equal>",     lambda _e: self.zoom_in())   # = / +
        self.root.bind("<plus>",      lambda _e: self.zoom_in())
        self.root.bind("<minus>",     lambda _e: self.zoom_out())
        self.root.bind("<KeyPress-0>", lambda _e: self.zoom_fit())

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
            raw = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
            self._corrections = {
                k: ([v] if isinstance(v, str) else v)
                for k, v in raw.items()
            }
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

    def _corrected_paths(self, src: Path) -> list[Path]:
        rel_src = src.relative_to(BASE_DIR).as_posix()
        rel_outs = self._corrections.get(rel_src)
        if rel_outs is None:
            p = OUT_DIR / src.relative_to(BASE_DIR)
            return [p] if p.exists() else []
        return [OUT_DIR / Path(r) for r in rel_outs if (OUT_DIR / Path(r)).exists()]

    def _is_corrected(self, src: Path) -> bool:
        return bool(self._corrected_paths(src))

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

        # Reset zoom and pan for each new image.
        self._zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.corrected = []
        for cp in self._corrected_paths(src):
            try:
                cdata = np.fromfile(str(cp), dtype=np.uint8)
                img = cv2.imdecode(cdata, cv2.IMREAD_COLOR)
                if img is not None:
                    self.corrected.append(img)
            except Exception:
                pass

        self._render_both()
        self._update_nav_label()
        self._set_load_status(src)

    def _set_load_status(self, src: Path) -> None:
        paths = self._corrected_paths(src)
        n_done = len(paths)
        n_boxes = len(self.points) // 4
        if n_done and self.points:
            self._set_status(
                f"{src.name}  [{n_done} box{'es' if n_done > 1 else ''} corrected]"
                " — adjust & re-apply, or Next ▶"
            )
        elif n_done:
            self._set_status(
                f"{src.name}  [{n_done} box{'es' if n_done > 1 else ''} corrected]"
                " — place corners to re-correct, or Next ▶"
            )
        elif n_boxes:
            self._set_status(f"{src.name} — {n_boxes} box(es) restored, adjust or Apply.")
        else:
            self._set_status(f"{src.name} — click 4 corners to start a box.")

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

    # ---------------------------------------------------------------- Zoom --

    def _apply_zoom(self, factor: float, cx: float | None = None, cy: float | None = None) -> None:
        """Update zoom state immediately; debounce the actual render to avoid rapid redraws."""
        new_zoom = max(ZOOM_MIN, min(self._zoom * factor, ZOOM_MAX))
        actual = new_zoom / self._zoom
        if cx is not None and cy is not None:
            self.pan_x = (cx - CANVAS_BUFFER) * (1 - actual) + self.pan_x * actual
            self.pan_y = (cy - CANVAS_BUFFER) * (1 - actual) + self.pan_y * actual
        self._zoom = new_zoom
        self.scale = self._base_scale * self._zoom
        if self._zoom_job is not None:
            self.root.after_cancel(self._zoom_job)
        self._zoom_job = self.root.after(100, self._render_image)

    def zoom_in(self) -> None:
        self._apply_zoom(ZOOM_STEP)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / ZOOM_STEP)

    def zoom_fit(self) -> None:
        if self._zoom_job is not None:
            self.root.after_cancel(self._zoom_job)
            self._zoom_job = None
        self._zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.scale = self._base_scale
        self._render_image()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self._apply_zoom(ZOOM_STEP, event.x, event.y)
        else:
            self._apply_zoom(1 / ZOOM_STEP, event.x, event.y)

    # ----------------------------------------------------------------- Pan --

    def _on_pan_start(self, event: tk.Event) -> None:
        self._panning = True
        self._pan_start = (event.x, event.y)
        self._pan_start_offset = (self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event: tk.Event) -> None:
        if not self._panning:
            return
        self.pan_x = self._pan_start_offset[0] + event.x - self._pan_start[0]
        self.pan_y = self._pan_start_offset[1] + event.y - self._pan_start[1]
        # Move the existing canvas item directly — no image re-encoding, no allocations.
        if self.canvas.find_withtag("img"):
            self.canvas.coords("img", CANVAS_BUFFER + self.pan_x, CANVAS_BUFFER + self.pan_y)
        self._redraw_points()

    def _on_pan_end(self, event: tk.Event) -> None:
        self._panning = False
        cursor = "fleur" if self._hit_test(event.x, event.y) is not None else "crosshair"
        self.canvas.config(cursor=cursor)

    # --------------------------------------------------------------- Render --

    def _render_both(self) -> None:
        self._render_image()
        self._render_corrected()

    def _render_image(self) -> None:
        self._zoom_job = None
        if self.original is None:
            return
        try:
            h, w = self.original.shape[:2]
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:   # canvas not yet laid out
                return

            usable_w = max(cw - 2 * CANVAS_BUFFER, 1)
            usable_h = max(ch - 2 * CANVAS_BUFFER, 1)
            self._base_scale = min(usable_w / w, usable_h / h, 1.0)
            self.scale = self._base_scale * self._zoom

            new_w = max(int(w * self.scale), 1)
            new_h = max(int(h * self.scale), 1)

            rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
            new_image = ImageTk.PhotoImage(pil)

            self.canvas.delete("all")
            self.tk_image = new_image   # assign after delete so old ref lives until now
            self.canvas.create_image(
                CANVAS_BUFFER + self.pan_x, CANVAS_BUFFER + self.pan_y,
                anchor=tk.NW, image=self.tk_image, tags="img",
            )
            self._redraw_points()
        except Exception:
            pass

    def _render_corrected(self) -> None:
        self.canvas_right.delete("all")
        cw = max(self.canvas_right.winfo_width(), 100)
        ch = max(self.canvas_right.winfo_height(), 100)

        if not self.corrected:
            self.canvas_right.create_text(
                cw // 2, ch // 2, text="No correction yet",
                fill="#666677", font=("Arial", 14),
            )
            return

        n = len(self.corrected)
        gap = 4
        slot_w = max((cw - gap * (n - 1)) // n, 1)

        pil_images: list[Image.Image] = []
        for img in self.corrected:
            ih, iw = img.shape[:2]
            s = min(slot_w / iw, ch / ih, 1.0)
            new_w = max(int(iw * s), 1)
            new_h = max(int(ih * s), 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))

        total_w = sum(p.width for p in pil_images) + gap * (n - 1)
        max_h   = max(p.height for p in pil_images)
        composite = Image.new("RGB", (total_w, max_h), (32, 34, 37))
        x = 0
        for pil in pil_images:
            composite.paste(pil, (x, (max_h - pil.height) // 2))
            x += pil.width + gap

        self.tk_corrected = ImageTk.PhotoImage(composite)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.tk_corrected)

        info = "  |  ".join(f"{img.shape[1]} × {img.shape[0]} px" for img in self.corrected)
        self.canvas_right.create_text(
            4, ch - 4, anchor=tk.SW, text=info, fill="#aaaaaa", font=("Arial", 9),
        )

    def _on_window_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._render_both)

    # ------------------------------------------------------- Point handling --

    def _boxes(self) -> list[list[tuple[float, float]]]:
        return [self.points[i:i + 4] for i in range(0, len(self.points), 4)]

    def _complete_boxes(self) -> list[list[tuple[float, float]]]:
        return [b for b in self._boxes() if len(b) == 4]

    def _to_canvas(self, ox: float, oy: float) -> tuple[float, float]:
        return (ox * self.scale + CANVAS_BUFFER + self.pan_x,
                oy * self.scale + CANVAS_BUFFER + self.pan_y)

    def _to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return ((cx - CANVAS_BUFFER - self.pan_x) / self.scale,
                (cy - CANVAS_BUFFER - self.pan_y) / self.scale)

    def _hit_test(self, cx: float, cy: float) -> int | None:
        best_idx, best_dist = None, float(HIT_RADIUS)
        for i, (ox, oy) in enumerate(self.points):
            px, py = self._to_canvas(ox, oy)
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
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
        ox, oy = self._to_image(event.x, event.y)
        self.points.append((ox, oy))
        self._redraw_points()
        n = len(self.points)
        remainder = n % 4
        box_num = (n - 1) // 4 + 1
        if remainder == 0:
            self._set_status(
                f"Box {box_num} complete — click to start box {box_num + 1}, or Apply."
            )
        else:
            self._set_status(f"Box {box_num}: {remainder}/4 points.")

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_idx is None or self.original is None:
            return
        ox, oy = self._to_image(event.x, event.y)
        self.points[self._drag_idx] = (ox, oy)
        self._redraw_points()

    def _on_release(self, _event: tk.Event) -> None:
        if self._drag_idx is not None:
            self._drag_idx = None
            n = len(self.points)
            if n > 0 and n % 4 == 0:
                b = n // 4
                self._set_status(
                    f"{b} box{'es' if b > 1 else ''} set — drag to adjust, or Apply."
                )

    def _on_hover(self, event: tk.Event) -> None:
        if self._panning:
            return
        cursor = "fleur" if self._hit_test(event.x, event.y) is not None else "crosshair"
        self.canvas.config(cursor=cursor)

    def reset_points(self) -> None:
        self.points.clear()
        self._redraw_points()
        if self.original is not None:
            self._set_status("Click 4 corners to start a box.")

    def _redraw_points(self) -> None:
        self.canvas.delete("overlay")
        if not self.points:
            return
        for box_idx, box in enumerate(self._boxes()):
            color = BOX_COLORS[box_idx % len(BOX_COLORS)]
            pts_c = [self._to_canvas(*p) for p in box]

            for i in range(len(pts_c) - 1):
                self.canvas.create_line(*pts_c[i], *pts_c[i + 1],
                                        fill=color, width=2, tags="overlay")
            if len(pts_c) == 4:
                self.canvas.create_line(*pts_c[-1], *pts_c[0],
                                        fill=color, width=2, tags="overlay")

            for pt_idx, (cx, cy) in enumerate(pts_c, start=1):
                self.canvas.create_oval(
                    cx - POINT_RADIUS, cy - POINT_RADIUS,
                    cx + POINT_RADIUS, cy + POINT_RADIUS,
                    fill=POINT_COLOR, outline=POINT_OUTLINE, width=2, tags="overlay",
                )
                self.canvas.create_text(
                    cx + 12, cy - 12, text=f"{box_idx + 1}.{pt_idx}",
                    fill=LABEL_COLOR, font=("Arial", 11, "bold"), tags="overlay",
                )

    # -------------------------------------------------------------- Apply ---

    def _filenames_for_folder(self, src: Path) -> list[str]:
        parent = src.relative_to(BASE_DIR).parent.as_posix()
        names: set[str] = set()
        for rel_src_key, rel_out_list in self._corrections.items():
            if Path(rel_src_key).parent.as_posix() == parent:
                for rel_out in rel_out_list:
                    names.add(Path(rel_out).name)
        return sorted(names)

    def _ask_save_options(self, src: Path, n_boxes: int) -> list[tuple[str, str]] | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Options")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        folder_names = self._filenames_for_folder(src)
        stem, suffix = src.stem, src.suffix

        label_combos: list[ttk.Combobox] = []
        name_combos:  list[ttk.Combobox] = []

        for i in range(n_boxes):
            if i > 0:
                ttk.Separator(dialog, orient=tk.HORIZONTAL).pack(
                    fill=tk.X, padx=12, pady=(6, 0)
                )
            header = f"Box {i + 1}" if n_boxes > 1 else "Save Options"
            ttk.Label(dialog, text=header, font=("Arial", 9, "bold")).pack(
                padx=16, pady=(10, 2), anchor="w"
            )
            ttk.Label(dialog, text="  Label  (blank = no sub-folder):").pack(
                padx=16, pady=(0, 2), anchor="w"
            )
            lc = ttk.Combobox(dialog, values=self._labels, width=36)
            lc.pack(padx=16, pady=(0, 6))
            label_combos.append(lc)

            ttk.Label(dialog, text="  Filename:").pack(padx=16, pady=(0, 2), anchor="w")
            default_name = src.name if i == 0 else f"{stem}_{i + 1}{suffix}"
            nc = ttk.Combobox(dialog, values=folder_names, width=36)
            nc.set(default_name)
            nc.pack(padx=16, pady=(0, 6))
            name_combos.append(nc)

        label_combos[0].focus_set()

        result: list[list[tuple[str, str]] | None] = [None]

        def on_ok(_event=None) -> None:
            out = []
            for i, (lc, nc) in enumerate(zip(label_combos, name_combos)):
                label    = lc.get().strip()
                filename = nc.get().strip()
                if not filename:
                    filename = src.name if i == 0 else f"{stem}_{i + 1}{suffix}"
                if not Path(filename).suffix:
                    filename += suffix
                out.append((label, filename))
            result[0] = out
            dialog.destroy()

        def on_cancel(_event=None) -> None:
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=(8, 14))
        ttk.Button(btn_frame, text="OK",     command=on_ok,     width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=6)

        all_combos = [c for pair in zip(label_combos, name_combos) for c in pair]
        for combo, nxt in zip(all_combos[:-1], all_combos[1:]):
            combo.bind("<Return>", lambda _e, n=nxt: n.focus_set())
        all_combos[-1].bind("<Return>", on_ok)
        dialog.bind("<Escape>", on_cancel)

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

        n = len(self.points)
        if n == 0 or n % 4 != 0:
            messagebox.showerror(
                "Invalid selection",
                f"Points must be a multiple of 4 (one complete box each).\n"
                f"Currently have {n} point{'s' if n != 1 else ''}."
                + ("\nFinish the current box or reset it." if n % 4 != 0 else ""),
            )
            return

        warped_list: list[np.ndarray] = []
        for box in self._complete_boxes():
            pts = np.array(box, dtype="float32")
            try:
                warped_list.append(four_point_transform(self.original, pts))
            except Exception as exc:
                messagebox.showerror("Transform failed", str(exc))
                return

        src = Path(self.image_path)
        options = self._ask_save_options(src, len(warped_list))
        if options is None:
            return

        rel_src = src.relative_to(BASE_DIR).as_posix()

        for old_rel in self._corrections.get(rel_src, []):
            old_file = OUT_DIR / Path(old_rel)
            if old_file.exists():
                old_file.unlink()
        if rel_src not in self._corrections:
            old_unlabeled = OUT_DIR / src.relative_to(BASE_DIR)
            if old_unlabeled.exists():
                old_unlabeled.unlink()

        new_rel_outs: list[str] = []
        rel_src_path = src.relative_to(BASE_DIR)

        for warped, (label, filename) in zip(warped_list, options):
            if label:
                rel_out = (rel_src_path.parent / label / filename).as_posix()
            else:
                rel_out = (rel_src_path.parent / filename).as_posix()
            out = OUT_DIR / Path(rel_out)
            out.parent.mkdir(parents=True, exist_ok=True)

            ext = out.suffix.lower() or ".png"
            ok, buf = cv2.imencode(ext, warped)
            if not ok:
                messagebox.showerror("Save failed", f"Could not encode {filename}.")
                return
            try:
                buf.tofile(str(out))
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return

            new_rel_outs.append(rel_out)
            if label and label not in self._labels:
                self._labels.append(label)
                self._labels.sort()

        self._save_labels()
        self._corrections[rel_src] = new_rel_outs
        self._save_corrections()

        self.corrected = warped_list
        self._render_corrected()
        self._update_nav_label()
        b = len(warped_list)
        self._set_status(f"Saved {b} box{'es' if b > 1 else ''} for {src.name}")

    # ------------------------------------------------------------- Helpers --

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
