"""Tkinter GUI for picking a 4-corner region and rectifying it."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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


class PerspectiveApp:
    """Main application window.

    Workflow:
        1. Open an image.
        2. Click the four corners of the text region (any order).
        3. Click "Apply & Save" to write a rectified PNG/JPG.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Perspective Correction Tool")

        # State
        self.image_path: str | None = None
        self.original: np.ndarray | None = None       # full-resolution BGR
        self.tk_image: ImageTk.PhotoImage | None = None
        self.scale: float = 1.0                        # display / original
        self.points: list[tuple[float, float]] = []    # in *original* coords
        self.last_warped: np.ndarray | None = None
        self._drag_idx: int | None = None              # index of point being dragged

        self._build_ui()
        # Re-fit image when the window is resized.
        self.root.bind("<Configure>", self._on_window_resize)
        self._resize_job: str | None = None

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Image…", command=self.open_image).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Reset Points", command=self.reset_points).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            toolbar, text="Apply && Save…", command=self.apply_and_save
        ).pack(side=tk.LEFT, padx=2)

        self.status = ttk.Label(toolbar, text="Open an image to begin.")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(
            self.root, bg="#202225", width=1000, height=700, cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_hover)

        # Helpful keyboard shortcuts
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-r>", lambda _e: self.reset_points())
        self.root.bind("<Return>", lambda _e: self.apply_and_save())

    # -------------------------------------------------------------- Loading --
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        # cv2.imread doesn't handle non-ASCII paths on Windows reliably;
        # use np.fromfile + imdecode for robustness.
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read file:\n{exc}")
            return
        if image is None:
            messagebox.showerror("Error", "Could not decode image.")
            return

        self.image_path = path
        self.original = image
        self.points.clear()
        self._render_image()
        self._set_status("Click the 4 corners of the text region (any order).")

    # --------------------------------------------------------------- Render --
    def _render_image(self) -> None:
        """Fit the original image into the current canvas size."""
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        self.scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = max(int(w * self.scale), 1), max(int(h * self.scale), 1)

        rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="img")
        self._redraw_points()

    def _on_window_resize(self, _event: tk.Event) -> None:
        # Debounce: only re-render after resizing settles for ~100 ms.
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._render_image)

    # ------------------------------------------------------- Point handling --
    def _hit_test(self, cx: float, cy: float) -> int | None:
        """Return the index of the nearest point within HIT_RADIUS, or None."""
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

        # Clicking near an existing point starts a drag instead of adding one.
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self._drag_idx = hit
            return

        if len(self.points) >= 4:
            self._set_status("Drag a point to reposition it, or 'Reset Points' to start over.")
            return

        ox = event.x / self.scale
        oy = event.y / self.scale
        self.points.append((ox, oy))
        self._redraw_points()

        if len(self.points) == 4:
            self._set_status("4 points set — drag to adjust, or click 'Apply & Save'.")
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
                self._set_status("4 points set — drag to adjust, or click 'Apply & Save'.")

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

        # Lines connecting points in click order, plus closing edge once full.
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

        # Numbered handles on top of the lines.
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

    # ---------------------------------------------------------- Apply/save --
    def apply_and_save(self) -> None:
        if self.original is None:
            messagebox.showinfo("No image", "Open an image first.")
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
        self.last_warped = warped

        # Suggest a filename next to the original.
        if self.image_path:
            base, _ext = os.path.splitext(os.path.basename(self.image_path))
            initial_dir = os.path.dirname(self.image_path)
            initial_file = f"{base}_corrected.png"
        else:
            initial_dir = os.getcwd()
            initial_file = "corrected.png"

        save_path = filedialog.asksaveasfilename(
            title="Save corrected image",
            defaultextension=".png",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("TIFF", "*.tif *.tiff")],
        )
        if save_path:
            ext = os.path.splitext(save_path)[1].lower() or ".png"
            ok, buf = cv2.imencode(ext, warped)
            if not ok:
                messagebox.showerror("Save failed", "Could not encode image.")
                return
            try:
                buf.tofile(save_path)  # unicode-path safe
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return
            self._set_status(f"Saved: {save_path}")

        self._show_preview(warped)

    def _show_preview(self, warped: np.ndarray) -> None:
        win = tk.Toplevel(self.root)
        win.title("Corrected Result")

        h, w = warped.shape[:2]
        max_dim = 900
        s = min(max_dim / w, max_dim / h, 1.0)
        disp_w, disp_h = max(int(w * s), 1), max(int(h * s), 1)

        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)

        label = tk.Label(win, image=photo, bg="#202225")
        label.image = photo  # keep a reference so it isn't garbage-collected
        label.pack(padx=8, pady=8)

        info = ttk.Label(win, text=f"Output size: {w} × {h} px")
        info.pack(pady=(0, 8))

    # ------------------------------------------------------------- Helpers --
    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
