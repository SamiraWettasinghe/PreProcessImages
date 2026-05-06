"""Tkinter GUI for picking a 4-corner region and rectifying it."""

from __future__ import annotations

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
HIT_RADIUS = 14  # canvas-px grab zone, larger than visual radius

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_HERE    = Path(__file__).resolve().parent
BASE_DIR = _HERE / "Akten_selektiert"
OUT_DIR  = _HERE / "Akten_selektiert_corrected"


class PerspectiveApp:
    """Main application window.

    Workflow:
        1. App loads all images from Akten_selektiert/ on startup.
        2. Click the four corners of the text region (any order); drag to adjust.
           The right pane shows the existing corrected version (if any).
        3. Click Apply — the rectified image is written to Akten_selektiert_corrected/
           mirroring the original subdirectory layout, and shown immediately on the right.
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

        self._resize_job: str | None = None
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

    def _output_path(self, src: Path) -> Path:
        return OUT_DIR / src.relative_to(BASE_DIR)

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

        # Load corrected version if it already exists.
        out = self._output_path(src)
        if out.exists():
            try:
                cdata = np.fromfile(str(out), dtype=np.uint8)
                self.corrected = cv2.imdecode(cdata, cv2.IMREAD_COLOR)
            except Exception:
                self.corrected = None
        else:
            self.corrected = None

        self._render_both()
        self._update_nav_label()
        self._set_load_status(src)

    def _set_load_status(self, src: Path) -> None:
        already_done = self._output_path(src).exists()
        if already_done and self.points:
            self._set_status(f"{src.name}  [corrected] — adjust & re-apply, or press Next ▶")
        elif already_done:
            self._set_status(f"{src.name}  [corrected] — click corners to re-correct, or press Next ▶")
        elif self.points:
            self._set_status(f"{src.name} — points restored, drag to adjust or click Apply.")
        else:
            self._set_status(f"{src.name} — click 4 corners of the text region.")

    def _update_nav_label(self) -> None:
        total = len(self._image_list)
        if total == 0:
            self._nav_label.config(text="— / —")
            return
        done = "✓" if self._output_path(self._image_list[self._current_idx]).exists() else " "
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

        self.scale = min(cw / w, ch / h, 1.0)
        new_w = max(int(w * self.scale), 1)
        new_h = max(int(h * self.scale), 1)

        rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
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

        info = f"{w} × {h} px"
        self.canvas_right.create_text(
            4, ch - 4, anchor=tk.SW, text=info,
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
            dx = ox * self.scale - cx
            dy = oy * self.scale - cy
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
        self.points.append((event.x / self.scale, event.y / self.scale))
        self._redraw_points()
        if len(self.points) == 4:
            self._set_status("4 points set — drag to adjust, or click Apply.")
        else:
            self._set_status(f"Selected {len(self.points)}/4 points.")

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_idx is None or self.original is None:
            return
        h, w = self.original.shape[:2]
        ox = max(0.0, min(event.x / self.scale, w - 1))
        oy = max(0.0, min(event.y / self.scale, h - 1))
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
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                self.canvas.create_line(
                    x1 * self.scale, y1 * self.scale,
                    x2 * self.scale, y2 * self.scale,
                    fill=LINE_COLOR, width=2, tags="overlay",
                )
        if len(self.points) == 4:
            x1, y1 = self.points[-1]
            x2, y2 = self.points[0]
            self.canvas.create_line(
                x1 * self.scale, y1 * self.scale,
                x2 * self.scale, y2 * self.scale,
                fill=LINE_COLOR, width=2, tags="overlay",
            )
        for i, (ox, oy) in enumerate(self.points, start=1):
            cx, cy = ox * self.scale, oy * self.scale
            self.canvas.create_oval(
                cx - POINT_RADIUS, cy - POINT_RADIUS,
                cx + POINT_RADIUS, cy + POINT_RADIUS,
                fill=POINT_COLOR, outline=POINT_OUTLINE, width=2, tags="overlay",
            )
            self.canvas.create_text(
                cx + 12, cy - 12, text=str(i),
                fill=LABEL_COLOR, font=("Arial", 12, "bold"), tags="overlay",
            )

    # ---------------------------------------------------------------- Apply --

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
        out = self._output_path(src)
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

        self.corrected = warped
        self._render_corrected()
        self._update_nav_label()
        self._set_status(f"Saved → {out.relative_to(_HERE)}")

    # ------------------------------------------------------------- Helpers --

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
