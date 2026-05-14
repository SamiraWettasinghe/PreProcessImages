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
ZOOM_STEP     = 1.25
ZOOM_MIN      = 0.25
ZOOM_MAX      = 20.0

BOX_COLORS = ["#22ff88", "#ff9922", "#22aaff", "#ff44cc", "#ffff44", "#aa44ff"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_HERE            = Path(__file__).resolve().parent
BASE_DIR         = _HERE / "Akten_selektiert"
OUT_DIR          = _HERE / "Akten_selektiert_corrected"
LABELS_FILE      = _HERE / "labels.json"
CORRECTIONS_FILE = _HERE / "corrections.json"
PREPROCESS_FILE  = _HERE / "preprocess_settings.json"


class _Tooltip:
    """Lightweight hover tooltip for any Tkinter widget."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 600) -> None:
        self._widget = widget
        self._text   = text
        self._delay  = delay
        self._win:  tk.Toplevel | None = None
        self._job:  str         | None = None
        widget.bind("<Enter>",       self._on_enter, add="+")
        widget.bind("<Leave>",       self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None) -> None:
        if self._job:
            self._widget.after_cancel(self._job)
        self._job = self._widget.after(self._delay, self._show)

    def _on_leave(self, _event=None) -> None:
        if self._job:
            self._widget.after_cancel(self._job)
            self._job = None
        if self._win:
            self._win.destroy()
            self._win = None

    def _show(self) -> None:
        self._job = None
        if self._win:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self._text, justify=tk.LEFT,
            background="#ffffcc", relief=tk.SOLID, borderwidth=1,
            wraplength=340, font=("Arial", 9), padx=6, pady=4,
        ).pack()


class PerspectiveApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Perspective Correction Tool")

        self.image_path: str | None = None
        self.original: np.ndarray | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self._base_scale: float = 1.0
        self._zoom: float = 1.0
        self.scale: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        self._panning: bool = False
        self._pan_start: tuple[int, int] = (0, 0)
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)

        self.points: list[tuple[float, float]] = []
        self._drag_idx: int | None = None

        self.corrected: list[np.ndarray] = []
        self.tk_corrected: ImageTk.PhotoImage | None = None

        self._image_list: list[Path] = []
        self._current_idx: int = 0
        self._saved_points: dict[str, list[tuple[float, float]]] = {}

        self._labels: list[str] = []
        self._corrections: dict[str, list[str]] = {}

        self._resize_job: str | None = None
        self._zoom_job:   str | None = None

        # Preprocessing state
        self._warped_raw: list[np.ndarray] = []
        self._pp_preview_job: str | None = None
        self._pp_panel:     ttk.LabelFrame | None = None
        self._pp_container: ttk.Frame      | None = None
        self._pp_save_btn:  ttk.Button     | None = None

        # Preprocessing parameters (defaults overwritten by saved settings)
        self._pp_enabled    = tk.BooleanVar(value=False)
        self._pp_denoise    = tk.BooleanVar(value=True)
        self._pp_denoise_h  = tk.IntVar(value=10)
        self._pp_clahe      = tk.BooleanVar(value=True)
        self._pp_clahe_clip = tk.DoubleVar(value=2.0)
        self._pp_clahe_grid = tk.IntVar(value=8)
        self._pp_thresh     = tk.StringVar(value="Adaptive")
        self._pp_block      = tk.IntVar(value=31)
        self._pp_c          = tk.IntVar(value=10)
        self._pp_morph      = tk.BooleanVar(value=False)
        self._pp_morph_k    = tk.IntVar(value=2)
        self._pp_upscale    = tk.BooleanVar(value=False)
        self._pp_upscale_f  = tk.DoubleVar(value=2.0)

        self._load_labels()
        self._load_corrections()
        self._load_preprocess_settings()
        self._build_ui()
        self.root.bind("<Configure>", self._on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        ttk.Checkbutton(
            toolbar, text="Preprocess",
            variable=self._pp_enabled,
            command=self._toggle_preprocess_panel,
        ).pack(side=tk.LEFT, padx=2)

        self._pp_save_btn = ttk.Button(
            toolbar, text="Save Preproc", state=tk.DISABLED,
            command=self._save_preprocess_only,
        )
        self._pp_save_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._done_label = ttk.Label(toolbar, text=" ", width=2)
        self._done_label.pack(side=tk.LEFT, padx=(4, 0))
        self._page_var = tk.StringVar()
        self._page_entry = ttk.Entry(toolbar, textvariable=self._page_var, width=5,
                                     justify=tk.RIGHT)
        self._page_entry.pack(side=tk.LEFT, padx=0)
        self._page_entry.bind("<Return>", self._on_jump)
        self._page_entry.bind("<FocusOut>", self._restore_page_entry)
        self._total_label = ttk.Label(toolbar, text="/ —")
        self._total_label.pack(side=tk.LEFT, padx=(2, 4))
        self.status = ttk.Label(toolbar, text="Loading…")
        self.status.pack(side=tk.LEFT, padx=8)

        # Collapsible preprocessing panel sits between toolbar and content.
        # The container is always packed (zero height when panel is hidden).
        self._pp_container = ttk.Frame(self.root)
        self._pp_container.pack(side=tk.TOP, fill=tk.X)
        self._pp_panel = self._build_preprocess_panel(self._pp_container)
        if self._pp_enabled.get():
            self._pp_panel.pack(fill=tk.X, padx=6, pady=(0, 4))

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

        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)

        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>",   self._on_mousewheel)
        self.canvas.bind("<Button-5>",   self._on_mousewheel)

        self.root.bind("<Control-r>", lambda _e: self.reset_points())
        self.root.bind("<Return>",    lambda _e: self.apply_correction())
        self.root.bind("<Left>",      lambda _e: self.go_prev())
        self.root.bind("<Right>",     lambda _e: self.go_next())
        self.root.bind("<equal>",     lambda _e: self.zoom_in())
        self.root.bind("<plus>",      lambda _e: self.zoom_in())
        self.root.bind("<minus>",     lambda _e: self.zoom_out())
        self.root.bind("<KeyPress-0>", lambda _e: self.zoom_fit())

    def _build_preprocess_panel(self, parent: ttk.Frame) -> ttk.LabelFrame:
        panel = ttk.LabelFrame(parent, text="Preprocessing Parameters", padding=(8, 4))

        def tip(widget: tk.Widget, text: str) -> None:
            _Tooltip(widget, text)

        def make_slider(par, label, var, from_, to, fmt="{:.0f}",
                        width=7, tooltip="") -> ttk.Frame:
            f = ttk.Frame(par)
            lbl = ttk.Label(f, text=label, anchor="e", width=width)
            lbl.pack(side=tk.LEFT)
            val_lbl = ttk.Label(f, width=6, anchor="w")

            def on_slide(*_):
                v = var.get()
                if isinstance(var, tk.IntVar):
                    v = int(round(v))
                    var.set(v)
                val_lbl.config(text=fmt.format(v))
                self._schedule_preview()

            scale = ttk.Scale(f, from_=from_, to=to, variable=var,
                               orient=tk.HORIZONTAL, length=110, command=on_slide)
            scale.pack(side=tk.LEFT, padx=(2, 2))
            val_lbl.config(text=fmt.format(var.get()))
            val_lbl.pack(side=tk.LEFT)
            if tooltip:
                for w in (lbl, scale, val_lbl):
                    tip(w, tooltip)
            return f

        # ── Row 0: Denoise | CLAHE ──────────────────────────────────────────
        row0 = ttk.Frame(panel)
        row0.pack(fill=tk.X, pady=2)

        cb = ttk.Checkbutton(row0, text="Denoise", variable=self._pp_denoise,
                              command=self._schedule_preview)
        cb.pack(side=tk.LEFT, padx=(0, 2))
        tip(cb, "Removes scanner grain and paper speckle.\n\n"
                "Enable this if your scan looks 'noisy' or has a rough texture. "
                "Safe to leave on for most documents — it runs before everything else.\n\n"
                "Default: on")

        make_slider(row0, "h:", self._pp_denoise_h, 1, 40, width=3,
                    tooltip="Denoising strength.\n\n"
                            "Higher = more aggressive removal, but very high values can soften "
                            "fine pen strokes or thin typewriter characters.\n\n"
                            "Raise to 20–30 only if grain is still clearly visible in the preview.\n\n"
                            "Default: 10").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(row0, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        cb = ttk.Checkbutton(row0, text="CLAHE", variable=self._pp_clahe,
                              command=self._schedule_preview)
        cb.pack(side=tk.LEFT, padx=(0, 2))
        tip(cb, "Evens out uneven lighting across the page.\n\n"
                "Useful for yellowed paper, folds, or scans where one corner looks "
                "darker than another. Works by boosting contrast in small local "
                "regions rather than the page as a whole.\n\n"
                "Default: on")

        make_slider(row0, "Clip:", self._pp_clahe_clip, 0.5, 8.0, fmt="{:.1f}",
                    width=5,
                    tooltip="Contrast boost strength.\n\n"
                            "Higher = darker areas get a stronger lift, but values above 4 "
                            "can create bright halos around dark ink.\n\n"
                            "Default: 2.0").pack(side=tk.LEFT, padx=(0, 4))

        make_slider(row0, "Grid:", self._pp_clahe_grid, 4, 32, width=5,
                    tooltip="Tile size for local contrast adjustment.\n\n"
                            "The page is divided into a grid and each tile is adjusted "
                            "independently. Smaller grid = finer adjustments, but can look "
                            "patchy on very uniform pages.\n\n"
                            "Try 4 for pages with heavy uneven staining.\n\n"
                            "Default: 8").pack(side=tk.LEFT)

        # ── Row 1: Threshold ────────────────────────────────────────────────
        row1 = ttk.Frame(panel)
        row1.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(row1, text="Threshold:", width=10)
        lbl.pack(side=tk.LEFT)
        tip(lbl, "Converts the image to pure black-and-white, which makes text sharper "
                 "and more readable for OCR.\n\n"
                 "• None – keep as greyscale (no conversion)\n"
                 "• Otsu – good for clean, evenly-lit typewritten pages; finds one "
                 "global cutoff between ink and paper\n"
                 "• Adaptive – best for handwriting or mixed pages; decides black vs "
                 "white separately for each small patch, handling uneven ink and lighting\n\n"
                 "Default: Adaptive")

        thresh_cb = ttk.Combobox(row1, textvariable=self._pp_thresh,
                                  values=["None", "Otsu", "Adaptive"],
                                  state="readonly", width=9)
        thresh_cb.pack(side=tk.LEFT, padx=(2, 12))
        thresh_cb.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())
        tip(thresh_cb, "Converts the image to pure black-and-white, which makes text sharper "
                       "and more readable for OCR.\n\n"
                       "• None – keep as greyscale (no conversion)\n"
                       "• Otsu – good for clean, evenly-lit typewritten pages\n"
                       "• Adaptive – best for handwriting, mixed pages, or uneven lighting\n\n"
                       "Default: Adaptive")

        make_slider(row1, "Block:", self._pp_block, 11, 101,
                    tooltip="Adaptive threshold only — patch size in pixels.\n\n"
                            "Each pixel looks at its neighbours within this square to decide "
                            "whether it should be black or white.\n\n"
                            "Too small (e.g. 11) → text can look patchy or broken.\n"
                            "Too large (e.g. 101) → background grey bleeds into text.\n\n"
                            "Increase if you see blotchy patches; decrease if thin strokes "
                            "are disappearing.\n\n"
                            "Default: 31").pack(side=tk.LEFT, padx=(0, 4))

        make_slider(row1, "C:", self._pp_c, 1, 30, width=3,
                    tooltip="Adaptive threshold only — darkness bias.\n\n"
                            "Shifts the black/white boundary within each patch:\n"
                            "• Raise it if the background is coming out grey\n"
                            "• Lower it if ink strokes are turning white and disappearing\n\n"
                            "Default: 10").pack(side=tk.LEFT)

        # ── Row 2: Morphology | Upscale ─────────────────────────────────────
        row2 = ttk.Frame(panel)
        row2.pack(fill=tk.X, pady=2)

        cb = ttk.Checkbutton(row2, text="Morphology", variable=self._pp_morph,
                              command=self._schedule_preview)
        cb.pack(side=tk.LEFT, padx=(0, 2))
        tip(cb, "Fills tiny gaps and breaks in letter strokes.\n\n"
                "Most useful for handwriting where the pen lifted mid-letter, leaving "
                "small disconnected pieces. Leave off for clean typewritten documents "
                "— it can merge letters that are close together.\n\n"
                "Default: off")

        make_slider(row2, "Kernel:", self._pp_morph_k, 1, 7,
                    tooltip="Gap-filling brush size in pixels.\n\n"
                            "1–2 closes hairline breaks without affecting letter shape.\n"
                            "3+ fills larger gaps but risks merging nearby letters or "
                            "thickening thin strokes noticeably.\n\n"
                            "Default: 2").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        cb = ttk.Checkbutton(row2, text="Upscale", variable=self._pp_upscale,
                              command=self._schedule_preview)
        cb.pack(side=tk.LEFT, padx=(0, 2))
        tip(cb, "Enlarges the image before OCR.\n\n"
                "Most OCR engines need at least 300 DPI to read text reliably. "
                "Enable this if your scans are small, look blurry when zoomed in, "
                "or if the OCR is missing or garbling characters.\n\n"
                "Default: off")

        make_slider(row2, "Factor:", self._pp_upscale_f, 1.0, 4.0, fmt="{:.1f}×",
                    tooltip="Enlargement multiplier.\n\n"
                            "2.0 doubles the image dimensions (4× the pixels) and is "
                            "enough for most low-resolution scans.\n\n"
                            "Only go above 2.0 if the scan was originally very small "
                            "(under 150 DPI). Higher values increase file size and "
                            "processing time with diminishing OCR benefit.\n\n"
                            "Default: 2.0×").pack(side=tk.LEFT)

        return panel

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

    def _load_preprocess_settings(self) -> None:
        try:
            d = json.loads(PREPROCESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for var, key in [
            (self._pp_enabled,    "enabled"),
            (self._pp_denoise,    "denoise"),
            (self._pp_denoise_h,  "denoise_h"),
            (self._pp_clahe,      "clahe"),
            (self._pp_clahe_clip, "clahe_clip"),
            (self._pp_clahe_grid, "clahe_grid"),
            (self._pp_thresh,     "thresh"),
            (self._pp_block,      "block"),
            (self._pp_c,          "c"),
            (self._pp_morph,      "morph"),
            (self._pp_morph_k,    "morph_k"),
            (self._pp_upscale,    "upscale"),
            (self._pp_upscale_f,  "upscale_f"),
        ]:
            if key in d:
                try:
                    var.set(d[key])
                except Exception:
                    pass

    def _save_preprocess_settings(self) -> None:
        d = {
            "enabled":    self._pp_enabled.get(),
            "denoise":    self._pp_denoise.get(),
            "denoise_h":  self._pp_denoise_h.get(),
            "clahe":      self._pp_clahe.get(),
            "clahe_clip": float(self._pp_clahe_clip.get()),
            "clahe_grid": self._pp_clahe_grid.get(),
            "thresh":     self._pp_thresh.get(),
            "block":      self._pp_block.get(),
            "c":          self._pp_c.get(),
            "morph":      self._pp_morph.get(),
            "morph_k":    self._pp_morph_k.get(),
            "upscale":    self._pp_upscale.get(),
            "upscale_f":  float(self._pp_upscale_f.get()),
        }
        try:
            PREPROCESS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_preprocess_settings()
        self.root.destroy()

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

        self._zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Clear session warp buffer when moving to a new image
        self._warped_raw = []
        if self._pp_save_btn is not None:
            self._pp_save_btn.config(state=tk.DISABLED)

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
            self._done_label.config(text=" ")
            self._page_var.set("—")
            self._total_label.config(text="/ —")
            return
        done = "✓" if self._is_corrected(self._image_list[self._current_idx]) else " "
        self._done_label.config(text=done)
        self._page_var.set(str(self._current_idx + 1))
        self._total_label.config(text=f"/ {total}")

    def _on_jump(self, _event=None) -> str:
        try:
            page = int(self._page_var.get())
        except ValueError:
            self._restore_page_entry()
            return "break"
        self._go_to(page - 1)
        self.canvas.focus_set()
        return "break"

    def _restore_page_entry(self, _event=None) -> None:
        if self._image_list:
            self._page_var.set(str(self._current_idx + 1))

    def go_prev(self) -> None:
        self._go_to(self._current_idx - 1)

    def go_next(self) -> None:
        self._go_to(self._current_idx + 1)

    # ---------------------------------------------------------------- Zoom --

    def _apply_zoom(self, factor: float, cx: float | None = None, cy: float | None = None) -> None:
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
            if cw < 10 or ch < 10:
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
            self.tk_image = new_image
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

    # ------------------------------------------ Preprocessing toggle/pipeline --

    def _toggle_preprocess_panel(self) -> None:
        if self._pp_panel is None:
            return
        if self._pp_enabled.get():
            self._pp_panel.pack(fill=tk.X, padx=6, pady=(0, 4))
            self._schedule_preview()
        else:
            self._pp_panel.pack_forget()
            # Revert right pane to raw perspective-corrected image
            if self._warped_raw:
                self.corrected = list(self._warped_raw)
                self._render_corrected()
        self._save_preprocess_settings()

    def _schedule_preview(self) -> None:
        """Debounce preprocessing so sliders don't trigger a render on every tick."""
        if self._pp_preview_job is not None:
            self.root.after_cancel(self._pp_preview_job)
        self._pp_preview_job = self.root.after(300, self._apply_preview)

    def _apply_preview(self) -> None:
        self._pp_preview_job = None
        if not self._warped_raw or not self._pp_enabled.get():
            return
        try:
            self.corrected = [self._preprocess(img) for img in self._warped_raw]
        except Exception:
            return
        self._render_corrected()
        self._save_preprocess_settings()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        result = gray

        if self._pp_denoise.get():
            result = cv2.fastNlMeansDenoising(result, h=int(self._pp_denoise_h.get()))

        if self._pp_clahe.get():
            grid = int(self._pp_clahe_grid.get())
            clahe = cv2.createCLAHE(
                clipLimit=float(self._pp_clahe_clip.get()),
                tileGridSize=(grid, grid),
            )
            result = clahe.apply(result)

        thresh = self._pp_thresh.get()
        if thresh == "Otsu":
            _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh == "Adaptive":
            block = int(self._pp_block.get())
            block = block if block % 2 == 1 else block + 1  # adaptive requires odd
            block = max(block, 3)
            result = cv2.adaptiveThreshold(
                result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, int(self._pp_c.get()),
            )

        if self._pp_morph.get():
            k = int(self._pp_morph_k.get())
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        if self._pp_upscale.get():
            factor = float(self._pp_upscale_f.get())
            h, w = result.shape[:2]
            result = cv2.resize(result, (int(w * factor), int(h * factor)),
                                 interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def _save_preprocess_only(self) -> None:
        """Re-save preprocessed output reusing the same paths, without re-warping."""
        if not self._warped_raw or self.image_path is None:
            return
        src = Path(self.image_path)
        rel_src = src.relative_to(BASE_DIR).as_posix()
        rel_outs = self._corrections.get(rel_src, [])
        if not rel_outs:
            messagebox.showinfo("Nothing to save",
                                 "Apply the perspective correction first.")
            return
        try:
            processed = [self._preprocess(w) for w in self._warped_raw]
        except Exception as exc:
            messagebox.showerror("Preprocessing failed", str(exc))
            return

        for proc_img, rel_out in zip(processed, rel_outs):
            out = OUT_DIR / Path(rel_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            ext = out.suffix.lower() or ".png"
            ok, buf = cv2.imencode(ext, proc_img)
            if ok:
                buf.tofile(str(out))

        self.corrected = processed
        self._render_corrected()
        n = len(processed)
        self._set_status(
            f"Saved {n} preprocessed image{'s' if n > 1 else ''} for {src.name}"
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

        # Store raw warped images for live re-preprocessing via sliders
        self._warped_raw = list(warped_list)

        to_save = (
            [self._preprocess(w) for w in warped_list]
            if self._pp_enabled.get()
            else warped_list
        )

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

        for save_img, (label, filename) in zip(to_save, options):
            if label:
                rel_out = (rel_src_path.parent / label / filename).as_posix()
            else:
                rel_out = (rel_src_path.parent / filename).as_posix()
            out = OUT_DIR / Path(rel_out)
            out.parent.mkdir(parents=True, exist_ok=True)

            ext = out.suffix.lower() or ".png"
            ok, buf = cv2.imencode(ext, save_img)
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

        self.corrected = to_save
        if self._pp_save_btn is not None:
            self._pp_save_btn.config(state=tk.NORMAL)
        self._render_corrected()
        self._update_nav_label()
        b = len(to_save)
        pp_note = " (preprocessed)" if self._pp_enabled.get() else ""
        self._set_status(f"Saved {b} box{'es' if b > 1 else ''}{pp_note} for {src.name}")

    # ------------------------------------------------------------- Helpers --

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
