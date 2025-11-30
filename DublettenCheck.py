from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView, QMessageBox, QAbstractItemView,
    QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt
from shiboken6 import isValid
from typing import List, Dict, Any, Optional
import sys


class GroupEditDialog(QDialog):
    """Dialog zur Bearbeitung einer einzelnen Dubletten-Gruppe."""

    def __init__(self, group_df: Any, column_mappings: Dict[str, str], parent=None):
        super().__init__(parent)
        self.group_df = group_df.copy()
        self.column_mappings = column_mappings
        self.setWindowTitle(f"Dubletten-Gruppe bearbeiten ({len(group_df)} Zeilen)")
        self.setMinimumWidth(900)
        self.setMinimumHeight(500)
        self._detail_window = None
        self.changes = {}
        self._init_ui()
        self._populate_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Titel der Gruppe (sicherer Zugriff)
        titel_col = self.column_mappings.get('titel', 'TitelNeu')
        titel_sample = str(self.group_df.iloc[0].get(titel_col, 'Unbekannter Titel'))[:80]
        
        title_label = QLabel(f"<h3>📝 Dubletten-Titel: {titel_sample}</h3>")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Info-Label
        info_label = QLabel(
            f"<i>{len(self.group_df)} Zeilen haben identische Titel. "
            "Sie können die Titel und andere Felder direkt in der Tabelle bearbeiten.</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Tabelle
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("✅ Änderungen übernehmen")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.accept)

        self.btn_cancel = QPushButton("❌ Abbrechen")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _populate_table(self):
        """Füllt die Tabelle mit den Daten der Gruppe."""
        columns = ['Index', 'Modul-IDs', 'Original-Titel', 'Neuer Titel (editierbar)', 'Einleitung (editierbar)', 'Inhalte (editierbar)']
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        # Header anpassen
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)

        self.table.setRowCount(len(self.group_df))

        for row_idx, (orig_idx, row) in enumerate(self.group_df.iterrows()):
            # Index (nicht bearbeitbar)
            idx_item = QTableWidgetItem(str(orig_idx))
            idx_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row_idx, 0, idx_item)

            # Modul-IDs (nicht bearbeitbar)
            mod_col = self.column_mappings.get('modul', 'Modulnummern')
            mod_text = str(row.get(mod_col, ''))[:50]
            mod_item = QTableWidgetItem(mod_text)
            mod_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row_idx, 1, mod_item)

            # Original-Titel (nicht bearbeitbar)
            orig_col = self.column_mappings.get('original', 'Neu zertifizierter Titel der Maßnahme NEU')
            orig_titel = str(row.get(orig_col, ''))
            orig_item = QTableWidgetItem(orig_titel)
            orig_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row_idx, 2, orig_item)

            # Neuer Titel (bearbeitbar)
            titel_col = self.column_mappings.get('titel', 'TitelNeu')
            new_titel = str(row.get(titel_col, ''))
            self.table.setItem(row_idx, 3, QTableWidgetItem(new_titel))

            # Einleitung (bearbeitbar)
            einl_col = self.column_mappings.get('einleitung', 'EinleitungNeu')
            einleitung = str(row.get(einl_col, ''))
            self.table.setItem(row_idx, 4, QTableWidgetItem(einleitung))

            # Inhalte (bearbeitbar)
            inh_col = self.column_mappings.get('inhalte', 'Inhalte_Feld')
            inhalte = str(row.get(inh_col, ''))
            self.table.setItem(row_idx, 5, QTableWidgetItem(inhalte[:200]))

    def get_changes(self) -> Dict[int, Dict[str, str]]:
        """Gibt die gemachten Änderungen zurück: {index: {feld: wert}}"""
        changes = {}

        for row_idx in range(self.table.rowCount()):
            orig_idx = int(self.table.item(row_idx, 0).text())
            row_changes = {}

            # Nur Felder prüfen, die geändert wurden
            original_row = self.group_df.loc[orig_idx]

            # Neuer Titel
            titel_col = self.column_mappings.get('titel', 'TitelNeu')
            new_titel = self.table.item(row_idx, 3).text()
            if new_titel != str(original_row.get(titel_col, '')):
                row_changes[titel_col] = new_titel

            # Einleitung
            einl_col = self.column_mappings.get('einleitung', 'EinleitungNeu')
            new_einleitung = self.table.item(row_idx, 4).text()
            if new_einleitung != str(original_row.get(einl_col, '')):
                row_changes[einl_col] = new_einleitung

            # Inhalte
            inh_col = self.column_mappings.get('inhalte', 'Inhalte_Feld')
            new_inhalte = self.table.item(row_idx, 5).text()
            if new_inhalte != str(original_row.get(inh_col, '')):
                row_changes[inh_col] = new_inhalte

            if row_changes:
                changes[orig_idx] = row_changes

        return changes


class DuplicateResolutionDialog(QDialog):
    """Hauptdialog zur Verwaltung aller Dubletten-Gruppen."""

    _app_instance = None

    def __init__(self, duplicate_groups: List[Any], column_mappings: Dict[str, str], parent=None):
        super().__init__(parent)
        self.duplicate_groups = duplicate_groups
        self.column_mappings = column_mappings
        self.all_changes = {}
        self.setWindowTitle(f"Dubletten-Auflösung ({len(duplicate_groups)} Gruppen)")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Hauptüberschrift
        title_label = QLabel("<h2>🔍 Dubletten-Prüfung</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Info-Label
        info_label = QLabel(
            f"<b>{len(self.duplicate_groups)} Gruppen mit identischen Titeln gefunden.</b><br>"
            "Doppelklicken Sie auf eine Gruppe, um die Titel zu bearbeiten:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Liste der Dubletten-Gruppen
        self.group_list = QListWidget()
        # WICHTIG: Korrekte Signal-Verbindung (nur item übergeben)
        self.group_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.group_list)

        # Gruppen in Liste einfügen
        titel_col = self.column_mappings.get('titel', 'TitelNeu')
        for idx, group in enumerate(self.duplicate_groups):
            titel = str(group.iloc[0].get(titel_col, 'Unbekannter Titel'))
            item_text = f"Gruppe {idx+1}: \"{titel[:60]}...\" → {len(group)} betroffene Zeilen"
            self.group_list.addItem(item_text)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_edit = QPushButton("✏️ Ausgewählte Gruppe bearbeiten")
        self.btn_edit.clicked.connect(self._edit_selected_group)
        self.btn_edit.setStyleSheet("background-color: #2196F3; color: white;")

        self.btn_skip = QPushButton("⏭️ Alle überspringen")
        self.btn_skip.clicked.connect(self.reject)

        self.btn_finish = QPushButton("✅ Änderungen übernehmen & Schließen")
        self.btn_finish.clicked.connect(self.accept)
        self.btn_finish.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_skip)
        btn_layout.addWidget(self.btn_finish)
        layout.addLayout(btn_layout)

    def _on_item_double_clicked(self, item):
        """Wird beim Doppelklick auf ein ListItem aufgerufen."""
        if item:
            # Finde den Index des geklickten Items
            row = self.group_list.row(item)
            if row >= 0:
                self._edit_group(row)

    def _edit_selected_group(self):
        """Bearbeitet die aktuell ausgewählte Gruppe."""
        current_row = self.group_list.currentRow()
        if current_row >= 0:
            self._edit_group(current_row)

    def _edit_group(self, group_idx):
        """Öffnet den Edit-Dialog für eine spezifische Gruppe (nur noch 1 Parameter!)."""
        group = self.duplicate_groups[group_idx]

        # WICHTIG: KEIN parent übergeben, um Sichtbarkeitsprobleme zu vermeiden
        edit_dialog = GroupEditDialog(group, self.column_mappings)
        result = edit_dialog.exec()

        if result == QDialog.Accepted:
            # Änderungen speichern
            changes = edit_dialog.get_changes()
            self.all_changes.update(changes)

            # Visuelles Feedback
            self.group_list.item(group_idx).setBackground(Qt.green)
            
        # Optional: Nach Bearbeitung automatisch zur nächsten Gruppe springen
        # self.group_list.setCurrentRow(group_idx + 1)

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
    def resolve_duplicates(duplicate_groups: List[Any], column_mappings: Dict[str, str]) -> Optional[Dict[int, Dict[str, str]]]:
        """Statische Methode zum Aufrufen des Dialogs."""
        print(f"🔍 UI wird initialisiert mit {len(duplicate_groups)} Gruppen...")
        
        try:
            # WICHTIG: Hier kein parent übergeben!
            app = DuplicateResolutionDialog.get_qt_app()
            print("✅ QApplication bereit")
            
            dialog = DuplicateResolutionDialog(duplicate_groups, column_mappings)
            print("✅ Dialog erstellt, zeige jetzt...")
            
            # SICHERSTELLEN, dass der Dialog sichtbar wird
            dialog.setModal(True)
            dialog.show()  # Explizit anzeigen
            
            result = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()

            print(f"✅ Dialog geschlossen mit Ergebnis: {result}")
            
            if result == QDialog.Accepted:
                print(f"✅ Änderungen gesammelt: {len(dialog.all_changes)} Einträge")
                return dialog.all_changes
            else:
                print("🚨 Benutzer hat abgebrochen oder übersprungen")
                return None
        
        except Exception as e:
            print(f"❌ FEHLER in resolve_duplicates: {e}")
            import traceback
            traceback.print_exc()
            return None


# === TEST-MODUS ===
if __name__ == '__main__':
    import pandas as pd
    
    print("🧪 Starte UI-Test...")
    
    # Beispiel-DataFrame
    test_df = pd.DataFrame({
        'Neuer Titel': ['Kurs A', 'Kurs A', 'Kurs B', 'Kurs B'],
        'Modulnummern': ['M001', 'M002', 'M003', 'M004'],
        'Neu zertifizierter Titel der Maßnahme NEU': ['Alt A1', 'Alt A2', 'Alt B1', 'Alt B2'],
        'EinleitungNeu': ['Einf A1', 'Einf A2', 'Einf B1', 'Einf B2'],
        'Inhalte_Feld': ['Inhalt A1', 'Inhalt A2', 'Inhalt B1', 'Inhalt B2']
    })
    
    # Finde Dubletten
    test_df['_titel_norm'] = test_df['Neuer Titel'].str.lower()
    duplicate_mask = test_df.duplicated(subset=['_titel_norm'], keep=False)
    groups = test_df[duplicate_mask].groupby('_titel_norm')
    duplicate_groups = [group.drop(columns=['_titel_norm']) for _, group in groups]
    
    print(f"🧪 Gefundene Dubletten-Gruppen: {len(duplicate_groups)}")
    
    # Test-Mapping
    test_mapping = {
        'titel': 'Neuer Titel',
        'original': 'Neu zertifizierter Titel der Maßnahme NEU',
        'modul': 'Modulnummern',
        'einleitung': 'EinleitungNeu',
        'inhalte': 'Inhalte_Feld'
    }
    
    # Öffne Dialog
    print("📂 Öffne UI-Dialog...")
    changes = DuplicateResolutionDialog.resolve_duplicates(duplicate_groups, test_mapping)
    print(f"🧪 Test abgeschlossen. Änderungen: {changes}")
    
    if changes:
        print("✅ UI funktioniert einwandfrei!")
    else:
        print("⚠️ UI wurde abgebrochen oder es gab keine Änderungen.")