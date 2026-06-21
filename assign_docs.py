"""Tkinter app for grouping a person's corrected crops into numbered documents.

A downstream consolidation step, run *after* the external ``_X`` merge: it shows every
corrected crop for one person at once (from the main ``Akten_selektiert_corrected/`` set),
lets you tag each with a document number and page order, then renames the files — keeping
``corrections.json`` in step — so each crop's name reveals its document and page.

Run with ``python assign_docs.py``. Pure logic lives in ``documents.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from merge_sources import (
    ARCHIVE_DIRNAME,
    BASE_DIRNAME,
    CORRECTED_DIRNAME,
    archive_move,
    group_box,
    normalize_original_key,
)
from documents import (
    CropEntry,
    apply_renames,
    list_people,
    plan_changes,
    scan_person_crops,
)

_HERE            = Path(__file__).resolve().parent
BASE_DIR         = _HERE / BASE_DIRNAME
OUT_DIR          = _HERE / CORRECTED_DIRNAME
ARCHIVE_DIR      = _HERE / ARCHIVE_DIRNAME
CORRECTIONS_FILE = _HERE / "corrections.json"
BOXES_FILE       = _HERE / "boxes.json"
LABELS_FILE      = _HERE / "labels.json"
DELETE_TAG       = "assign_deleted"   # archive sub-folder for crops removed here

THUMB_W, THUMB_H = 175, 235
GRID_COLUMNS     = 4
PREVIEW_DIM      = 0.30   # brightness kept outside the crop region in the context view
PREVIEW_OUTLINE  = (40, 220, 255)   # BGR — crop outline in the context view
HOVER_DELAY      = 350    # ms before a hover preview appears
HOVER_W, HOVER_H = 460, 620   # max size of the floating hover preview

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "lang_title":     "Language",
        "lang_prompt":    "Choose a language:",
        "window_title":   "Document Assignment Tool",
        "person":         "Person:",
        "btn_prev":       "◀ Prev",
        "btn_next":       "Next ▶",
        "btn_apply":      "Apply / Rename",
        "no_people":      "No corrected crops found in {dir}.",
        "card_doc":       "Doc #",
        "card_page":      "Page #",
        "card_type":      "Type",
        "card_delete":    "Delete",
        "del_confirm_title": "Delete crop",
        "del_confirm_msg": "Move this crop to the archive?\n\n{name}\n\n"
                          "It will be moved to {dir}/{tag}/ (recoverable), and removed "
                          "from corrections.json and boxes.json.",
        "del_failed_title": "Delete failed",
        "del_failed_msg":  "Could not archive {name}: {exc}",
        "unassigned":     "(unassigned)",
        "preview_hint":   "(hover to preview · click for context)",
        "preview_title":  "Context — {name}",
        "preview_none":   "Original scan / crop region unavailable; showing the crop.",
        "grouping_title": "Documents (in page order)",
        "no_assignments": "No crops assigned to a document yet.",
        "nothing_title":  "Nothing to do",
        "nothing_msg":    "No filename changes are needed.",
        "confirm_title":  "Rename crops",
        "confirm_msg":    "{n} crop(s) will be renamed. Proceed?",
        "done_title":     "Done",
        "done_msg":       "{n} crop(s) renamed.",
        "unlinked_msg":   "\n\n{n} renamed crop(s) had no corrections.json entry "
                          "(file renamed, but not referenced).",
        "ok":             "OK",
        "cancel":         "Cancel",
    },
    "de": {
        "lang_title":     "Sprache",
        "lang_prompt":    "Sprache wählen:",
        "window_title":   "Dokumentzuordnung",
        "person":         "Person:",
        "btn_prev":       "◀ Zurück",
        "btn_next":       "Weiter ▶",
        "btn_apply":      "Übernehmen / Umbenennen",
        "no_people":      "Keine korrigierten Ausschnitte in {dir} gefunden.",
        "card_doc":       "Dok. Nr.",
        "card_page":      "Seite Nr.",
        "card_type":      "Typ",
        "card_delete":    "Löschen",
        "del_confirm_title": "Ausschnitt löschen",
        "del_confirm_msg": "Diesen Ausschnitt ins Archiv verschieben?\n\n{name}\n\n"
                          "Er wird nach {dir}/{tag}/ verschoben (wiederherstellbar) und "
                          "aus corrections.json und boxes.json entfernt.",
        "del_failed_title": "Löschen fehlgeschlagen",
        "del_failed_msg":  "{name} konnte nicht archiviert werden: {exc}",
        "unassigned":     "(nicht zugeordnet)",
        "preview_hint":   "(überfahren für Vorschau · klicken für Kontext)",
        "preview_title":  "Kontext — {name}",
        "preview_none":   "Originalscan / Ausschnittsbereich nicht verfügbar; "
                          "zeige den Ausschnitt.",
        "grouping_title": "Dokumente (in Seitenreihenfolge)",
        "no_assignments": "Noch keine Ausschnitte einem Dokument zugeordnet.",
        "nothing_title":  "Nichts zu tun",
        "nothing_msg":    "Keine Dateinamen müssen geändert werden.",
        "confirm_title":  "Ausschnitte umbenennen",
        "confirm_msg":    "{n} Ausschnitt(e) werden umbenannt. Fortfahren?",
        "done_title":     "Fertig",
        "done_msg":       "{n} Ausschnitt(e) umbenannt.",
        "unlinked_msg":   "\n\n{n} umbenannte Ausschnitt(e) hatten keinen Eintrag in "
                          "corrections.json (Datei umbenannt, aber nicht referenziert).",
        "ok":             "OK",
        "cancel":         "Abbrechen",
    },
}


def decode_thumb(path: Path, max_w: int, max_h: int,
                 upscale: bool = False) -> ImageTk.PhotoImage | None:
    """Non-ASCII-safe thumbnail (np.fromfile + cv2.imdecode, as in app.py).

    With ``upscale=True`` small images are enlarged to fill the box (for hover previews);
    otherwise they are only ever shrunk.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        s = min(max_w / w, max_h / h) if upscale else min(max_w / w, max_h / h, 1.0)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (max(int(w * s), 1), max(int(h * s), 1)), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)
    except Exception:
        return None


class DocAssignApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.withdraw()
        self.lang = self._ask_language()
        self.root.deiconify()

        self._corrections: dict[str, list[str]] = {}
        self._load_corrections()
        self._boxes_by_key: dict[str, list[tuple[float, float]]] = {}
        self._load_boxes()
        self._labels: list[str] = []
        self._load_labels()

        self._people = list_people(OUT_DIR)
        self._person_idx = 0
        self._entries: list[CropEntry] = []
        # Per-card widget state: list of dicts {entry, doc_var, page_var, thumb}.
        self._rows: list[dict] = []
        # Floating hover-preview state.
        self._hover_after: str | None = None
        self._hover_win: tk.Toplevel | None = None
        self._hover_img = None

        self._build_ui()
        if not self._people:
            messagebox.showinfo(
                self.t("window_title"),
                self.t("no_people", dir=OUT_DIR.name),
            )
        else:
            self._load_person(0)

    # ----------------------------------------------------------------- i18n --

    def t(self, key: str, **kw) -> str:
        s = TRANSLATIONS[self.lang].get(key, key)
        return s.format(**kw) if kw else s

    def _ask_language(self) -> str:
        dialog = tk.Toplevel(self.root)
        dialog.title(TRANSLATIONS["en"]["lang_title"])
        dialog.resizable(False, False)
        dialog.grab_set()
        choice = {"lang": "en"}
        ttk.Label(dialog, text=TRANSLATIONS["en"]["lang_prompt"],
                  font=("Arial", 11)).pack(padx=28, pady=(20, 14))
        btns = ttk.Frame(dialog)
        btns.pack(padx=28, pady=(0, 20))

        def pick(code: str) -> None:
            choice["lang"] = code
            dialog.destroy()

        ttk.Button(btns, text="English", width=14,
                   command=lambda: pick("en")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Deutsch", width=14,
                   command=lambda: pick("de")).pack(side=tk.LEFT, padx=6)
        dialog.protocol("WM_DELETE_WINDOW", lambda: pick("en"))
        dialog.bind("<Return>", lambda _e: pick(choice["lang"]))
        dialog.update_idletasks()
        w, h = dialog.winfo_width(), dialog.winfo_height()
        sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
        dialog.wait_window()
        return choice["lang"]

    # -------------------------------------------------------------- persist --

    def _load_corrections(self) -> None:
        try:
            raw = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
            self._corrections = {
                k: ([v] if isinstance(v, str) else v) for k, v in raw.items()
            }
        except Exception:
            self._corrections = {}
        self._build_reverse()

    def _build_reverse(self) -> None:
        """corrected_rel -> (original_key, box_index) — locates a crop's source scan/box."""
        self._reverse: dict[str, tuple[str, int]] = {}
        for orig_key, outs in self._corrections.items():
            for i, rel in enumerate(outs):
                self._reverse[rel] = (orig_key, i)

    def _save_corrections(self) -> None:
        CORRECTIONS_FILE.write_text(
            json.dumps(self._corrections, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_boxes(self) -> None:
        """boxes.json is keyed by the original's absolute path; index by portable key.

        The raw dict (with its original keys) is kept so deletions can be written back.
        """
        try:
            self._boxes_raw = json.loads(BOXES_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._boxes_raw = {}
        self._reindex_boxes()

    def _reindex_boxes(self) -> None:
        self._boxes_by_key = {
            normalize_original_key(k): [tuple(p) for p in v]
            for k, v in self._boxes_raw.items()
        }

    def _save_boxes(self) -> None:
        BOXES_FILE.write_text(
            json.dumps(self._boxes_raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_labels(self) -> None:
        try:
            self._labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._labels = []

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.root.title(self.t("window_title"))

        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(toolbar, text=self.t("person")).pack(side=tk.LEFT, padx=(0, 4))
        self._person_var = tk.StringVar()
        self._person_combo = ttk.Combobox(
            toolbar, textvariable=self._person_var, values=self._people,
            state="readonly", width=32,
        )
        self._person_combo.pack(side=tk.LEFT)
        self._person_combo.bind("<<ComboboxSelected>>", self._on_person_pick)

        ttk.Button(toolbar, text=self.t("btn_prev"),
                   command=self.go_prev).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(toolbar, text=self.t("btn_next"),
                   command=self.go_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text=self.t("btn_apply"),
                   command=self.apply).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Left: scrollable thumbnail grid.
        left = ttk.Frame(body)
        self._canvas = tk.Canvas(left, highlightthickness=0)
        vbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._grid = ttk.Frame(self._canvas)
        self._grid_win = self._canvas.create_window((0, 0), window=self._grid, anchor="nw")
        self._grid.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._grid_win, width=e.width),
        )
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)
        body.add(left, weight=3)

        # Right: live grouped readout.
        right = ttk.Labelframe(body, text=self.t("grouping_title"))
        self._grouping = tk.Text(right, width=34, wrap="word", state="disabled")
        gbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self._grouping.yview)
        self._grouping.configure(yscrollcommand=gbar.set)
        gbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._grouping.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        body.add(right, weight=1)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # -------------------------------------------------------------- person --

    def _on_person_pick(self, _event=None) -> None:
        name = self._person_var.get()
        if name in self._people:
            self._load_person(self._people.index(name))

    def go_prev(self) -> None:
        if self._people:
            self._load_person((self._person_idx - 1) % len(self._people))

    def go_next(self) -> None:
        if self._people:
            self._load_person((self._person_idx + 1) % len(self._people))

    def _load_person(self, idx: int) -> None:
        self._person_idx = idx
        person = self._people[idx]
        self._person_var.set(person)
        self._entries = scan_person_crops(OUT_DIR, person)
        self._build_cards()
        self._refresh_grouping()
        self._canvas.yview_moveto(0.0)

    def _build_cards(self) -> None:
        self._hover_hide()
        for child in self._grid.winfo_children():
            child.destroy()
        self._rows = []
        for c in range(GRID_COLUMNS):
            self._grid.columnconfigure(c, weight=1)

        for i, e in enumerate(self._entries):
            card = ttk.Frame(self._grid, relief=tk.RIDGE, borderwidth=1, padding=4)
            card.grid(row=i // GRID_COLUMNS, column=i % GRID_COLUMNS,
                      padx=4, pady=4, sticky="n")

            thumb = decode_thumb(OUT_DIR / e.rel_out, THUMB_W, THUMB_H)
            img_lbl = ttk.Label(card, image=thumb, cursor="hand2")
            img_lbl.image = thumb   # keep a reference so it isn't GC'd
            img_lbl.pack()
            img_lbl.bind("<Button-1>", lambda _ev, en=e: self._show_context(en))
            img_lbl.bind("<Enter>", lambda _ev, en=e: self._hover_schedule(en))
            img_lbl.bind("<Leave>", lambda _ev: self._hover_hide())

            label_txt = e.base_stem + e.ext
            ttk.Label(card, text=label_txt, wraplength=THUMB_W,
                      justify="center", font=("Arial", 8)).pack(pady=(2, 0))
            ttk.Label(card, text=self.t("preview_hint"), foreground="#888",
                      font=("Arial", 7)).pack(pady=(0, 4))

            row_frame = ttk.Frame(card)
            row_frame.pack()
            ttk.Label(row_frame, text=self.t("card_type")).grid(row=0, column=0, sticky="e")
            type_var = tk.StringVar(value=e.type_label)
            type_combo = ttk.Combobox(row_frame, textvariable=type_var, width=5,
                                      values=self._type_choices())
            type_combo.grid(row=0, column=1, padx=(2, 0))
            ttk.Label(row_frame, text=self.t("card_doc")).grid(row=1, column=0, sticky="e")
            doc_var = tk.StringVar(value="" if e.doc is None else str(e.doc))
            doc_combo = ttk.Combobox(row_frame, textvariable=doc_var, width=5)
            doc_combo.grid(row=1, column=1, padx=(2, 0))
            ttk.Label(row_frame, text=self.t("card_page")).grid(row=2, column=0, sticky="e")
            page_var = tk.StringVar(value="" if e.page is None else str(e.page))
            page_spin = ttk.Spinbox(row_frame, from_=1, to=999, width=5,
                                    textvariable=page_var)
            page_spin.grid(row=2, column=1, padx=(2, 0))

            for var in (doc_var, page_var):
                var.trace_add("write", lambda *_: self._refresh_grouping())

            ttk.Button(card, text=self.t("card_delete"),
                       command=lambda en=e: self._delete_crop(en)).pack(pady=(4, 0))

            self._rows.append({
                "entry": e, "doc_var": doc_var, "page_var": page_var,
                "type_var": type_var, "doc_combo": doc_combo,
            })
        self._refresh_doc_choices()

    def _type_choices(self) -> list[str]:
        """Classification suggestions: known labels plus types present for this person."""
        present = {e.type_label for e in self._entries if e.type_label}
        return [""] + sorted(set(self._labels) | present)

    # ----------------------------------------------------------- assigning --

    @staticmethod
    def _as_int(s: str) -> int | None:
        s = s.strip()
        try:
            return int(s)
        except ValueError:
            return None

    def _collect_desired(self) -> dict[str, tuple[int | None, int | None, str]]:
        """rel_out -> (doc|None, page|None, type_label) for every card."""
        out: dict[str, tuple[int | None, int | None, str]] = {}
        for row in self._rows:
            doc = self._as_int(row["doc_var"].get())
            if doc is None or doc <= 0:
                doc = None
            page = (self._as_int(row["page_var"].get()) or 1) if doc is not None else None
            out[row["entry"].rel_out] = (doc, page, row["type_var"].get().strip())
        return out

    def _refresh_doc_choices(self) -> None:
        """Offer the document numbers already in use as combobox suggestions."""
        docs = sorted({d for d, _p, _t in self._collect_desired().values() if d is not None})
        values = [str(d) for d in docs]
        for row in self._rows:
            row["doc_combo"].configure(values=values)

    def _refresh_grouping(self) -> None:
        self._refresh_doc_choices()
        desired = self._collect_desired()
        by_rel = {e.rel_out: e for e in self._entries}
        docs: dict[int, list[str]] = {}
        for rel_out, (doc, _page, _typ) in desired.items():
            if doc is not None:
                docs.setdefault(doc, []).append(rel_out)
        for doc in docs:
            docs[doc].sort(key=lambda r: (desired[r][1] if desired[r][1] is not None else 0, r))

        self._grouping.configure(state="normal")
        self._grouping.delete("1.0", tk.END)
        if not docs:
            self._grouping.insert(tk.END, self.t("no_assignments"))
        else:
            for doc in sorted(docs):
                self._grouping.insert(tk.END, f"D{doc:02d}\n")
                for page, rel_out in enumerate(docs[doc], start=1):
                    name = by_rel[rel_out].base_stem + by_rel[rel_out].ext
                    self._grouping.insert(tk.END, f"   p{page:02d}  {name}\n")
                self._grouping.insert(tk.END, "\n")
        self._grouping.configure(state="disabled")

    def apply(self) -> None:
        desired = self._collect_desired()
        renames = plan_changes(self._entries, desired, OUT_DIR)
        if not renames:
            messagebox.showinfo(self.t("nothing_title"), self.t("nothing_msg"))
            return
        if not messagebox.askokcancel(self.t("confirm_title"),
                                      self.t("confirm_msg", n=len(renames))):
            return
        self._corrections, unlinked = apply_renames(
            OUT_DIR, self._corrections, renames)
        self._save_corrections()
        self._build_reverse()   # corrected_rel keys changed; keep context lookup valid
        msg = self.t("done_msg", n=len(renames))
        if unlinked:
            msg += self.t("unlinked_msg", n=len(unlinked))
        messagebox.showinfo(self.t("done_title"), msg)
        self._load_person(self._person_idx)   # reload so names reflect the rename

    # -------------------------------------------------------------- deleting --

    def _delete_crop(self, entry: CropEntry) -> None:
        """Archive one crop (recoverable) and drop it from corrections.json / boxes.json."""
        name = entry.base_stem + entry.ext
        if not messagebox.askyesno(
            self.t("del_confirm_title"),
            self.t("del_confirm_msg", name=name, dir=ARCHIVE_DIR.name, tag=DELETE_TAG),
            icon="warning", default="no",
        ):
            return

        abs_file = OUT_DIR / entry.rel_out
        try:
            archive_move(abs_file, ARCHIVE_DIR, DELETE_TAG, entry.rel_out)
        except Exception as exc:
            messagebox.showerror(self.t("del_failed_title"),
                                 self.t("del_failed_msg", name=name, exc=exc))
            return

        rev = self._reverse.get(entry.rel_out)
        if rev:
            orig_key, idx = rev
            outs = self._corrections.get(orig_key)
            if outs and 0 <= idx < len(outs):
                del outs[idx]
                if outs:
                    self._corrections[orig_key] = outs
                else:
                    del self._corrections[orig_key]
                self._save_corrections()
            self._remove_box(orig_key, idx)
            self._build_reverse()

        # Prune the type sub-folder if the archive move emptied it.
        parent = abs_file.parent
        if parent != OUT_DIR and parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass

        self._load_person(self._person_idx)

    def _remove_box(self, orig_key: str, idx: int) -> None:
        """Drop the idx-th box (4 points) for an original from boxes.json, if present."""
        raw_key = next(
            (k for k in self._boxes_raw if normalize_original_key(k) == orig_key), None)
        if raw_key is None:
            return
        pts = self._boxes_raw[raw_key]
        start = idx * 4
        if start + 4 <= len(pts):
            del pts[start:start + 4]
        if pts:
            self._boxes_raw[raw_key] = pts
        else:
            del self._boxes_raw[raw_key]
        self._reindex_boxes()
        self._save_boxes()

    # ----------------------------------------------------------- hover view --

    def _hover_schedule(self, entry: CropEntry) -> None:
        self._hover_hide()
        self._hover_after = self.root.after(
            HOVER_DELAY, lambda: self._hover_show(entry))

    def _hover_hide(self) -> None:
        if self._hover_after is not None:
            try:
                self.root.after_cancel(self._hover_after)
            except Exception:
                pass
            self._hover_after = None
        if self._hover_win is not None:
            self._hover_win.destroy()
            self._hover_win = None
            self._hover_img = None

    def _hover_show(self, entry: CropEntry) -> None:
        """Borderless floating preview of the crop near the pointer.

        The window is positioned *before* it is mapped: an overrideredirect window
        otherwise flashes at the screen origin first, which sits under the pointer for the
        top-left thumbnail and ping-pongs Enter/Leave. The popup is also offset clear of
        the pointer so moving within the same thumbnail never lands on it.
        """
        self._hover_after = None
        img = decode_thumb(OUT_DIR / entry.rel_out, HOVER_W, HOVER_H, upscale=True)
        if img is None:
            return
        win = tk.Toplevel(self.root)
        win.withdraw()                       # stay hidden until placed
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        frame = tk.Frame(win, background="#000", bd=1)
        frame.pack()
        tk.Label(frame, image=img, background="#222").pack()
        self._hover_img = img   # keep a reference
        self._hover_win = win

        # Size from the image (no mapping needed), so we can place before showing.
        w, h = img.width() + 2, img.height() + 2   # +2 for the 1px frame border
        px, py = self.root.winfo_pointerxy()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        x = px + 24 if px + 24 + w <= sw else px - w - 24
        y = py + 16 if py + 16 + h <= sh else py - h - 16
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h))
        win.geometry(f"+{x}+{y}")
        win.deiconify()

    # ----------------------------------------------------------- context view --

    def _show_context(self, entry: CropEntry) -> None:
        """Pop up a large view: the source scan dimmed, with this crop's region lit."""
        self._hover_hide()
        orig_path = None
        quad = None
        rev = self._reverse.get(entry.rel_out)
        if rev:
            orig_key, idx = rev
            pts = self._boxes_by_key.get(orig_key)
            quad = group_box(pts, idx) if pts else None
            cand = BASE_DIR / orig_key
            if cand.exists():
                orig_path = cand

        photo, note = self._render_context(orig_path, quad, entry)
        if photo is None:
            return

        win = tk.Toplevel(self.root)
        win.title(self.t("preview_title", name=entry.base_stem + entry.ext))
        win.configure(background="#222")
        lbl = tk.Label(win, image=photo, cursor="hand2", background="#222")
        lbl.image = photo   # keep a reference
        lbl.pack()
        if note:
            tk.Label(win, text=note, foreground="#ffb0b0", background="#222").pack(pady=4)
        win.bind("<Escape>", lambda _e: win.destroy())
        lbl.bind("<Button-1>", lambda _e: win.destroy())
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        x = (sw - win.winfo_width()) // 2
        y = max((sh - win.winfo_height()) // 2, 0)
        win.geometry(f"+{x}+{y}")
        win.transient(self.root)

    def _render_context(self, orig_path, quad, entry):
        """Return (PhotoImage, note). Falls back to the enlarged crop if no scan/region."""
        max_w = int(self.root.winfo_screenwidth() * 0.9)
        max_h = int(self.root.winfo_screenheight() * 0.85)

        def load_bgr(path):
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                return None

        note = ""
        img = load_bgr(orig_path) if orig_path is not None else None
        if img is not None and quad is not None:
            poly = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
            dark = (img.astype(np.float32) * PREVIEW_DIM).astype(np.uint8)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            res = np.where(mask[:, :, None].astype(bool), img, dark)
            thick = max(2, img.shape[1] // 350)
            cv2.polylines(res, [poly], True, PREVIEW_OUTLINE, thick, cv2.LINE_AA)
        else:
            res = load_bgr(OUT_DIR / entry.rel_out)
            note = self.t("preview_none")
            if res is None:
                return None, None

        h, w = res.shape[:2]
        s = min(max_w / w, max_h / h)   # enlarge small images too
        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (max(int(w * s), 1), max(int(h * s), 1)), Image.LANCZOS)
        return ImageTk.PhotoImage(pil), note


def main() -> None:
    root = tk.Tk()
    root.geometry("1500x860")
    DocAssignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
