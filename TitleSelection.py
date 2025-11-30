from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QLineEdit, QPushButton, QHBoxLayout, QMessageBox
)
from TitleSelectionDetailDialog import DetailDialog
from PySide6.QtCore import Qt
from shiboken6 import isValid
import sys
from typing import List


class TitleSelectionDialog(QDialog):
    """Dialog zur Auswahl **und Bearbeitung** eines Titels."""

    _app_instance = None

    def __init__(self, title: str, titles: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Titel-Auswahl")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        
        # Variable wird später mit Informationen für Titel befüllt
        self._detail_data = None
        self._detail_window = None

        # Original-Titel (nur Anzeige)
        label_orig = QLabel(f"<b>Original:</b><br>{title}")
        label_orig.setWordWrap(True)
        layout.addWidget(label_orig)

        # Bearbeitbares Feld – immer mit aktuellem Titel gefüllt
        self.line_edit = QLineEdit()
        self.line_edit.setClearButtonEnabled(True)
        layout.addWidget(self.line_edit)

        # Liste aller Varianten
        self.list_widget = QListWidget()
        for t in titles:
            self.list_widget.addItem(QListWidgetItem(t))
        self.list_widget.setCurrentRow(0)          # erstes Element vorselektieren
        self.line_edit.setText(titles[0])          # Edit-Feld mit erstem Titel füllen
        layout.addWidget(self.list_widget)

        # Signal: Bei Auswahl -> Edit-Feld aktualisieren
        self.list_widget.currentItemChanged.connect(self._update_edit)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Auswählen")
        self.btn_cancel = QPushButton("Abbrechen")
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        self.btn_details = QPushButton("Details anzeigen …")
        btn_layout.insertWidget(1, self.btn_details)   # zwischen Auswählen & Abbrechen
        self.btn_details.clicked.connect(self._show_details)

        # Aktionen
        self.btn_select.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    # --------------------------------------------------
    # Hilfs-Methoden
    # --------------------------------------------------
    def _update_edit(self, current: QListWidgetItem, _):
        """Füllt das Edit-Feld mit dem gerade gewählten Titel."""
        if current:
            self.line_edit.setText(current.text())

    # --------------------------------------------------
    # Öffentliche Hilfs-Methoden
    # --------------------------------------------------
    @classmethod
    def get_qt_app(cls):
        if cls._app_instance is None:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app.setQuitOnLastWindowClosed(False)
            cls._app_instance = app
        return cls._app_instance

    @staticmethod
    def get_selected_title(
        original_title: str,
        titles: List[str],
        detail_data: dict | None = None          # <-- neu
    ) -> str | None:
        """Zeigt Dialog und gibt den bearbeiteten Text zurück."""
        app = TitleSelectionDialog.get_qt_app()
        dialog = TitleSelectionDialog(original_title, titles)
        if detail_data:                          # <-- setzen, falls vorhanden
            dialog.set_detail_data(detail_data)
        result = dialog.exec()
        return dialog.line_edit.text().strip() if result == QDialog.Accepted else None
    
    def set_detail_data(self, data: dict):
        """
        data = {
            'kursuebersicht': str,
            'original_title': str,
            'inhalte': str,
            'best_kw': str
        }
        """
        self._detail_data = data
    
    # ---------- Details ----------
    def _show_details(self):
        if not self._detail_data:
            QMessageBox.information(self, "Hinweis", "Keine Detail-Daten vorhanden.")
            return
        # Falls schon offen – einfach in den Vordergrund
        # WICHTIG: isValid() prüfen, sonst Fehler nach Schließen!
        if (self._detail_window is not None and 
            isValid(self._detail_window) and 
            self._detail_window.isVisible()):
            self._detail_window.raise_()
            self._detail_window.activateWindow()
            return
        self._detail_window = DetailDialog(self._detail_data, parent=self)
        self._detail_window.show()          # nicht exec() – nicht-modal!

    # ---------- beim Akzeptieren ----------
    def accept(self):
        if (self._detail_window is not None and
            isValid(self._detail_window) and          # <- C++-Objekt noch da?
            self._detail_window.isVisible()):
            self._detail_window.close()
        super().accept()