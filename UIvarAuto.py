from __future__ import annotations
import sys
from typing import List, Tuple, Optional, Dict
import pandas as pd
import re
from difflib import SequenceMatcher
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QLineEdit, QPushButton, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from shiboken6 import isValid

# Deine bestehenden Imports
from TitleSelectionDetailDialog import DetailDialog

# Annahme: Deine bestehenden Helper-Funktionen sind irgendwo verfügbar
# Falls nicht, müssen diese entsprechend importiert werden:
# from your_module import normalize_text, to_number


class KeywordSelectionDialog(QDialog):
    """
    Modal-Dialog: Liste mit Keyword-Vorschlägen + frei editierbares Feld.
    Rückgabe: nur das Keyword (str) – egal ob aus Liste oder selbst getippt.
    NEU: Live-Suche in keywords_df mit Debouncing.
    """

    _app: Optional[QApplication] = None

    def __init__(
        self,
        title: str,
        rows: List[Tuple[str, float, float, float, float]],
        keywords_df: pd.DataFrame = None,           # NEU: DataFrame für Suche
        search_threshold: float = 0.3,              # NEU: Mindest-Ähnlichkeit
        search_top_n: int = 25,                     # NEU: Max Ergebnisse
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Keyword Auswahl")
        self.Title = f"{title}"
        self.resize(650, 450)

        # --- Daten ---
        self._rows = rows                          # Original-Tupel (kw, m, v, r, t)
        self._original_rows = rows                 # Backup der Original-Rows
        self._current_rows = rows                  # Aktuell angezeigte Rows
        self._detail_data = None
        self._detail_window = None
        
        # --- NEU: Such-Parameter ---
        self._keywords_df = keywords_df
        self._search_threshold = search_threshold
        self._search_top_n = search_top_n
        
        # --- NEU: Debouncing Timer ---
        self._search_timer = QTimer(self)
        self._search_timer.setInterval(300)  # 300ms Debouncing
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._perform_search)

        # --- UI ---
        self._setup_ui()

        # --- Verbindungen ---
        self._setup_connections()

        # Initiale Selektion
        #self.list_widget.setCurrentRow(0)

    # ------------------------------------------------------------------
    def _setup_ui(self):
        """Einrichten der UI-Elemente"""
        lay = QVBoxLayout(self)
    
        # --- Titel setzen, Kopfzeile
        self._title_label = QLabel(f"<b>Titel:</b><br>{self.Title}")
        self._title_label.setAlignment(Qt.AlignCenter)
        font = self._title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self._title_label.setFont(font)
        lay.addWidget(self._title_label)

        # Edit-Feld mit Such-Button
        edit_layout = QHBoxLayout()
        self.line_edit = QLineEdit()
        self.line_edit.setClearButtonEnabled(True)  # Standard X-Button
        
        # self.btn_search = QPushButton("🔍 Suchen")
        # self.btn_search.setEnabled(self._keywords_df is not None)
        # self.btn_search.setToolTip("Suche nach ähnlichen Keywords")
        
        edit_layout.addWidget(self.line_edit)
        # edit_layout.addWidget(self.btn_search)
        lay.addLayout(edit_layout)

        # Liste
        self.list_widget = QListWidget()
        lay.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Auswählen")
        self.btn_details = QPushButton("Details anzeigen …")
        self.btn_cancel = QPushButton("Abbrechen")

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_details)
        btn_layout.addWidget(self.btn_cancel)
        lay.addLayout(btn_layout)
        
        self._repopulate_list()  # Verwende neue Methode

    # ------------------------------------------------------------------
    def _setup_connections(self):
        """Verbindungen herstellen"""
        self.list_widget.currentRowChanged.connect(self._update_edit)
        self.btn_select.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_details.clicked.connect(self._show_details)

        # NEU: Such-Verhalten
        self.line_edit.textChanged.connect(self._on_search_text_changed)
        # self.btn_search.clicked.connect(self._perform_search)

    # ------------------------------------------------------------------
    # NEU: Such-Funktionalität
    # ------------------------------------------------------------------
    
    @staticmethod
    def _normalize_text(s: str) -> str:
        if not isinstance(s, str):
            raise TypeError(f"_normalize_text() erwartet str, bekam {type(s)}: {s!r}")
        s = s.lower()
        s = re.sub(r"[.../:\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    @staticmethod
    def _to_number(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        val = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(val)
        except ValueError:
            return 0.0

    
    def _on_search_text_changed(self, text: str):
        """Wird bei jeder Textänderung aufgerufen"""
        if not text.strip():
            # Feld ist leer → Original-Liste sofort wieder anzeigen
            self._show_original_list()
            self._search_timer.stop()
        else:
            # Debounced Suche starten
            self._search_timer.start()

    def _show_original_list(self):
        """Zeigt die Original-Liste wieder an"""
        self._current_rows = self._original_rows
        self._repopulate_list()

    def _perform_search(self):
        """Führt die Suche durch und aktualisiert die Liste"""
        if self._keywords_df is None:
            return

        search_term = self.line_edit.text().strip()
        if not search_term:
            self._show_original_list()
            return

        # Suche nach ähnlichen Keywords
        similar_results = self._find_similar_keywords(search_term)

        # Konvertiere zu Rows-Format
        if similar_results:
            self._current_rows = [
                (r['keyword'], r['similarity'], r['vol'], r['results'], r['total'])
                for r in similar_results
            ]
        else:
            self._current_rows = []

        # Aktualisiere UI
        self._repopulate_list()

    def _find_similar_keywords(self, search_term: str) -> List[Dict]:
        """
        Findet ähnliche Keywords in keywords_df
        Kombiniert String-Ähnlichkeit und Token-Matching parallel
        """
        if self._keywords_df is None or not search_term.strip():
            return []

        # Normalize search term
        search_norm = self._normalize_text(search_term)
        search_tokens = set(search_norm.split())

        if not search_tokens:
            return []

        results = []

        # Volume Map (aus deiner Original-Funktion)
        volume_map = {
            "very low": 0.0,
            "low": 0.33,
            "medium": 0.66,
            "high": 1.0
        }

        # Max results für Normalisierung
        max_results = self._keywords_df["# Suchergebnisse mein Now"].apply(self._to_number).max()

        for _, row in self._keywords_df.iterrows():
            keyword = row['Suchwort']
            keyword_norm = self._normalize_text(keyword)
            keyword_tokens = set(keyword_norm.split())

            # String-Ähnlichkeit (character-level)
            char_similarity = SequenceMatcher(None, search_norm, keyword_norm).ratio()

            # Token-Ähnlichkeit
            token_similarity = self._calculate_token_similarity(search_tokens, keyword_tokens)

            # Kombiniere beide: maximale Ähnlichkeit (parallel)
            combined_similarity = max(char_similarity, token_similarity)

            # Muss über Threshold sein
            if combined_similarity < self._search_threshold:
                continue

            # Metriken berechnen (nur basierend auf Suchbegriff)
            vol = volume_map.get(str(row.get('Kategorie Suchvolumen', '')).lower(), 0.0)

            # Results Score
            results_val = self._to_number(row.get("# Suchergebnisse mein Now", 0))
            results_score = results_val / max_results if max_results else 0.0

            # Total Score (vereinfacht: 70% Ähnlichkeit, 10% Vol, 20% Results)
            total_score = 0.7 * combined_similarity + 0.1 * vol + 0.2 * results_score

            results.append({
                'keyword': keyword,
                'similarity': combined_similarity,
                'vol': vol,
                'results': results_score,
                'total': total_score
            })

        # Sortiere nach Total Score
        results.sort(key=lambda x: x['total'], reverse=True)

        # Begrenze auf top_n
        return results[:self._search_top_n]

    def _calculate_token_similarity(self, tokens1: set, tokens2: set) -> float:
        """Berechnet Jaccard-Ähnlichkeit zwischen Token-Sets"""
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    # ------------------------------------------------------------------
    # Angepasste Bestehende Methoden
    # ------------------------------------------------------------------
    def _repopulate_list(self):
        """Neu befüllen der Liste (ersetzt _populate_list)"""
        self.list_widget.blockSignals(True)  # Verbindungen blockieren
        self.list_widget.clear()

        if not self._current_rows:
            # Leere Liste → zeige Info-Item
            QListWidgetItem("Keine Keywords gefunden.", self.list_widget)
            self.btn_select.setEnabled(False)
            return

        self.btn_select.setEnabled(True)

        # Max Breite für Padding berechnen
        fm = QFontMetrics(self.list_widget.font())
        max_px = max((fm.horizontalAdvance(kw) for kw, *_ in self._current_rows), default=0)

        for kw, m, v, r, t in self._current_rows:
            # Berechne Padding
            pad = " " * max(0, int((max_px - fm.horizontalAdvance(kw)) / fm.horizontalAdvance(" ")))
            txt = f"{kw}{pad} | Ähnlichkeit: {m:.2f} | Vol: {v:.2f} | Ergebnisse: {r:.2f} | Total: {t:.3f}"
            QListWidgetItem(txt, self.list_widget)
            
        self.list_widget.blockSignals(False)  # Verbindungen wieder aktivieren

    def _update_edit(self, row: int) -> None:
        """Edit-Feld nur mit reinem Keyword füllen"""
        if 0 <= row < len(self._current_rows):
            self.line_edit.setText(self._current_rows[row][0])

    # ------------------------------------------------------------------
    # Details-Methoden (unverändert)
    # ------------------------------------------------------------------
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

    def accept(self):
        """Beim Akzeptieren Detail-Fenster schließen.
         WICHTIG: Timer stoppen, um QTimer warnings zu vermeiden!"""
        self._search_timer.stop()
        
        if (self._detail_window is not None and
            isValid(self._detail_window) and
            self._detail_window.isVisible()):
            self._detail_window.close()
        super().accept()
    
    def reject(self):
        """Beim Abbrechen Timer stoppen"""
        self._search_timer.stop()
        super().reject()

    # ------------------------------------------------------------------
    @staticmethod
    def get_selected_keyword(
        title: str,
        rows: List[Tuple[str, float, float, float, float]],
        keywords_df: pd.DataFrame = None,           # NEU
        search_threshold: float = 0.3,              # NEU
        search_top_n: int = 25,                     # NEU
        parent=None,
        detail_data: dict | None = None             # <-- bestandteil
    ) -> Optional[str]:
        """Statische Convenience-Methode – liefert nur das Keyword (str)."""
        app = QApplication.instance() or QApplication(sys.argv)
        dlg = KeywordSelectionDialog(
            title, rows,
            keywords_df=keywords_df,
            search_threshold=search_threshold,
            search_top_n=search_top_n,
            parent=parent
        )
        if detail_data:                          # <-- setzen, falls vorhanden
            dlg.set_detail_data(detail_data)
        return dlg.line_edit.text().strip() if dlg.exec() == QDialog.Accepted else None