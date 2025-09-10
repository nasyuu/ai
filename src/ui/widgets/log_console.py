# src/ui/widgets/log_console.py
from __future__ import annotations

import logging
import tkinter as tk
from tkinter.scrolledtext import ScrolledText


class TkLogHandler(logging.Handler):
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget
        self.widget.configure(state=tk.DISABLED)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.widget.after(0, self._append, msg + "\n")

    def _append(self, text: str):
        self.widget.configure(state=tk.NORMAL)
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state=tk.DISABLED)
