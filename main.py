"""Entry point: launches the perspective-correction GUI."""

import tkinter as tk

from app import PerspectiveApp


def main() -> None:
    root = tk.Tk()
    root.geometry("1200x820")
    PerspectiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
