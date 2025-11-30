# FloX / VarAuto — Zusammenfassung & Bedienungsanleitung

## 🚀 Überblick

**FloX (VarAuto)** ist ein lokales Python-/PyInstaller-Tool zur automatisierten Verarbeitung und Optimierung von Kursvariationen, Keywords, Modulkombinationen und KI-gestützter Textgenerierung.
Es unterstützt Excel-basierte Arbeitsprozesse (ESTHER, Edtelligent, Kursangebote, Mapping) und schreibt die Ergebnisse automatisiert in die Variationstabelle zurück.

Das Tool benötigt **keine technischen Kenntnisse** von Endanwendern – lediglich die korrekten Dateien im Ordner und die korrekte Config.

---

# 📁 Benötigte Dateien

Alle **7 Dateien müssen im selben Ordner** liegen wie die `config.json`:

1. **Config-Datei** (mit absoluten Pfaden & Spaltenbezeichnungen)
2. **Variationstabelle** (Variationen / Modulkombi)
3. **Keywords-Tabelle** (Edtelligent)
4. **Kursangebote-Tabelle** (ESTHER)
5. **Mapping-Tabelle** (Einzelmodule)
6. **prompts.json**
7. **systematik.json**

Backups werden automatisch erzeugt (`*_backup.xlsx`, `*_progress.xlsx`).

---

# 🧩 Pflicht-Spalten laut Config

Die exakten Spaltenüberschriften stehen in der geladenen Config.
Beispiel (gekürzt):

```json
"columns": {
  "Modulnummern": "Modulnummern kommagetrennt NEU\nNach Übernahme ",
  "Kursübersicht": "Kursübersicht der Einzelmodule\nrot = ...",
  "EinleitungNeu": "Einleitung NEU Sabine ...",
  "TitelNeu": "Titeloptimierung ...",
  "Systematik_Feld": "Systematik in Kursnet",
  "Keyword_Feld": "Keyword-Optimierung ...",
  "Termine_Feld": "Termine_Feld",
  "Inhalte_Feld": "Inhalt NEU",
  "DauerFeld": "Dauer in Tagen",
  "Zielgruppe": "Zielgruppe",
  "Voraussetzungen": "Voraussetzungen",
  "Abschlussart": "Abschlussart",
  "Abschlussbezeichnung": "Abschlussbezeichnung"
}
```

⚠️ **Die Spaltenüberschriften müssen exakt übereinstimmen** – inklusive Leerzeichen, Sonderzeichen, `\n` usw.

---

# 🛠 Installation & Voraussetzungen

## 1. Python (nur für Entwickler)

* Python **3.11.7**
* spaCy Modell **de_core_news_md**
* FastText Modell **lid.176.ftz**

## 2. Endanwender

Nur die `.exe` und die 7 Dateien in einem Ordner – keine Installation notwendig.

## 3. Build (für Entwickler)

```bat
python -m venv venv
venv\Scripts\activate
python -m spacy download de_core_news_md

pyinstaller --onefile --name FloX --noconsole --icon=icon.ico ^
  --add-data "lid.176.ftz;." ^
  --collect-all spacy ^
  --collect-all de_core_news_md ^
  varAuto.py
```

---

# ▶️ Bedienung

## **1. Dateien in gemeinsamen Ordner ablegen**

Alle 7 Pflichtdateien + die EXE.

## **2. EXE starten**

`FloX.exe` starten → GUI öffnet sich.

## **3. Workflows**

Je nach Prozess:

* **Titel / Keywords generieren**
* **Systematik optimieren**
* **Inhalte neu erzeugen**
* **Abschlussbezeichnung optimieren**
* **Termine generieren**
* **Kompletten Workflow durchlaufen**

## **4. Ergebnisse**

* werden in die **Variationstabelle geschrieben**
* Backups werden automatisch erzeugt

---

# ⚙️ Wichtige technische Hinweise

### 🔧 OpenAI API Key / Guthaben prüfen

1. [https://platform.openai.com](https://platform.openai.com)
2. Rechts oben Profil → **Billing**

   * Guthaben einsehen
   * Auto-Recharge aktivieren
3. Neuer API Key:
   Profil → **API Keys** → „Create new secret key“

---

# ❗ Typische Fehler & Lösungen

### 1️⃣ **Spaltenmapping fehlerhaft**

* Ursache: Spaltenname weicht minimal ab
* Lösung: In Excel **genau aus Config kopieren**, inklusive `\n`

---

### 2️⃣ **prompts.json hat falsche Platzhalter**

* Prüfen: Jeder Placeholder muss im Code gesetzt und an `run_ai_prompt()` übergeben werden.

---

### 3️⃣ **Termine-GUI findet Spalten nicht**

* Momentan **hardcodierte Strings** im Code → nicht umbenennen

---

### 4️⃣ **sys.stderr is None**

Bei PyInstaller + `--noconsole`.
→ Logging auf Datei umleiten (bereits implementiert).

---

### 5️⃣ **Abschlussbezeichnung zu stark gekürzt**

→ Prompt anpassen.

---

### 6️⃣ **Kursnummern ohne Komma**

→ UI sollte Bearbeitungsdialog öffnen.

---

# 📌 Limitierungen & Zukunft (Roadmap)

### **To-Dos**

1. Spaltenmapping robuster machen
2. Sheet-Name Auswahl in GUI ergänzen
3. Azure-Anbindung fertigstellen
4. `create_termine` modularisieren
5. Neue Felder sauber mit korrektem dtype anlegen
6. Prompts flexibel von Speicherort laden/speichern
7. Timeout bei `wait_for_gui_result`
8. Kursnummern-Parsing robust machen
9. Abschlussbezeichnung Prompt verbessern

### **Optionale Verbesserungen**

* KI-Inhalte in mehrere Prompts aufteilen
* Keyword-Vorauswahl optimieren
* "Zurück"-Knopf für Dialoge
* Titel/Keyword zuerst komplett wählen, dann Felder generieren
* Lade-Statusfenster
* Einheitliche Großschreibung (ITIL, EDV etc.)
* Stopword-Liste in UI einbauen

---

# 📄 Lizenz

Nur intern verwendbar (Amadeus Fire AG).

