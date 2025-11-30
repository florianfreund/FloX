from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QScrollArea, QWidget
)
from PySide6.QtCore import Qt
import re


class ModuleEditDialog(QDialog):
    """
    Zeigt alle nicht gefundenen Modulnummern in einem Fenster.
    Jede Zeile: "Nicht gefundenes Modul Nummer: ..." + Eingabefeld (vorgefüllt).
    Bei 'Übernehmen' wird ein dict { idx: [eingegebene Nummern] } zurückgegeben.
    Wenn eine Zeile leer bleibt → wird als „abgebrochen“ behandelt (None).
    """
    def __init__(self, unfound_rows: list[tuple[int, list[str]]], parent=None):
        """
        unfound_rows: Liste von (idx, variants)
        Es wird nur eine Variante angezeigt (bevorzugt mit '/').
        """
        super().__init__(parent)
        self.setWindowTitle("Korrektur nicht gefundener Modulnummern")
        self.resize(600, 400)
        self._rows = unfound_rows

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Bitte korrigieren Sie die folgenden Modulnummern.\n"
            "Mehrere Nummern können durch Komma oder Leerzeichen getrennt werden.\n"
            "Leeres Feld = Zeile überspringen."
        ))

        # Scrollbarer Bereich für viele Zeilen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)

        self._edits = {}

        for idx, variants in self._rows:
            # Eine Variante auswählen (bevorzugt mit '/')
            if isinstance(variants, str):
                display_variant = variants.strip().upper()
            elif variants:
                display_variant = next((v for v in variants if "/" in v), variants[0]).strip().upper()
            else:
                display_variant = ""

            # Label und Eingabefeld
            lbl = QLabel(f"Nicht gefundenes Modul Nummer {idx}: {display_variant}")
            lbl.setWordWrap(True)
            edit = QLineEdit(display_variant)
            edit.setPlaceholderText("Korrigierte Modulnummer(n) eingeben …")
            vbox.addWidget(lbl)
            vbox.addWidget(edit)
            self._edits[idx] = edit

        container.setLayout(vbox)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self.btn_ok = QPushButton("Alle übernehmen")
        self.btn_cancel = QPushButton("Abbrechen")
        btn_bar.addWidget(self.btn_ok)
        btn_bar.addWidget(self.btn_cancel)
        layout.addLayout(btn_bar)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def results(self):
        """Gibt dict { idx -> [Liste der eingegebenen IDs] } oder None (bei Abbruch) zurück."""
        if self.exec() != QDialog.DialogCode.Accepted:
            return None

        out = {}
        for idx, edit in self._edits.items():
            text = edit.text().strip()
            if not text:
                out[idx] = None
                continue
            # IDs aus dem Text extrahieren (z. B. A805/68 oder A805-68)
            found = [m.strip().upper() for m in re.findall(r"[A-Z]?\d{3,4}[-/]?\d{1,4}", text, re.I)]
            out[idx] = found
        return out
