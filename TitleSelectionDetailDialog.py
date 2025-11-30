from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QLabel, QScrollArea, QWidget, QPushButton, QPlainTextEdit
)
from PySide6.QtCore import Qt


class DetailDialog(QDialog):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kurs-Details")
        self.resize(600, 700)
        self._best_kw = data.get("best_kw", "").strip()
        # Nicht-modales Fenster – Parent bleibt bedienbar
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose)  # automatisch löschen beim Schließen

        # --- Scrollbereich ---
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)

        # --- Styling ---
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                padding-top:8px; margin-top:8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left:10px; padding:0 4px;
            }
            QLabel#inhalte {
                font-family: "Consolas","Courier New",monospace;
                font-size:10pt; background:#fdfdfd;
                padding:6px; border:1px solid #ccc;
            }
        """)

        # --- Kursübersicht ---
        gb_kurs = QGroupBox("Kursübersicht")
        v1 = QVBoxLayout()
        lbl = QLabel(data.get('kursuebersicht', '-'))
        lbl.setWordWrap(True)          # <-- Zeilenumbruch erlauben
        lbl.setMinimumWidth(500)       # <-- oder passende Breite (optional)
        v1.addWidget(lbl)
        gb_kurs.setLayout(v1)
        lay.addWidget(gb_kurs)

        # --- Original-Titel ---
        gb_orig = QGroupBox("Zertifizierter Original-Titel")
        v2 = QVBoxLayout()
        v2.addWidget(QLabel(data.get('original_title', '-')))
        gb_orig.setLayout(v2)
        lay.addWidget(gb_orig)
        
        # --- Master Keyword ---
        best_kw = data.get('best_kw', '').strip()
        if best_kw:
            gb_kw = QGroupBox("Haupt-Keyword")
            v_kw = QVBoxLayout()
            v_kw.addWidget(QLabel(self._best_kw))
            gb_kw.setLayout(v_kw)
            lay.addWidget(gb_kw)

        # --- Inhalte ---
        gb_inh = QGroupBox("Inhalte")
        v3 = QVBoxLayout()
        txt = data.get('inhalte', '').strip() or '-'
        pte = QPlainTextEdit(txt)
        pte.setReadOnly(True)
        pte.setMaximumHeight(400)
        v3.addWidget(pte)
        gb_inh.setLayout(v3)
        lay.addWidget(gb_inh)

        lay.addStretch()

        # --- Schließen-Button ---
        btn = QPushButton("Schließen")
        btn.clicked.connect(self.close)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)
        outer.addWidget(btn)