@echo off
REM ============================================
REM 🔥 ZWINGT Arbeitsverzeichnis = Skript-Ordner
REM ============================================
pushd "%~dp0"
set "PROJECT_DIR=%CD%"
echo ==================================================
echo 🔧 VARAUTO FINAL BUILD
echo ==================================================
echo 📍 PROJEKT-ORDNER: %PROJECT_DIR%
echo.

REM SICHERHEITS-CHECK: Verhindert Bauen im Python-Systemordner
echo %PROJECT_DIR% | findstr /C:"Python311\Scripts" >nul
if %ERRORLEVEL% == 0 (
    echo ❌❌❌ FEHLER: Du bist im PYTHON-SYSTEMORDNER!
    echo    Bitte erstelle einen eigenen Projektordner.
    pause
    exit /b 1
)

REM Prüfe VENV
if not exist "%PROJECT_DIR%\venv\Scripts\activate" (
    echo ❌ VENV fehlt! Erstelle es mit:
    echo    cd "%PROJECT_DIR%"
    echo    python -m venv venv
    pause
    exit /b 1
)

REM Aktiviere VENV
call "%PROJECT_DIR%\venv\Scripts\activate"
echo ✅ VENV aktiviert

REM Prüfe spaCy-Modell direkt (zuverlässiger)
echo.
echo 📂 Prüfe spaCy-Modell...
python -c "import spacy; spacy.load('de_core_news_md'); print('  ✅ Modell geladen')"
if errorlevel 1 (
    echo ❌ spaCy Modell fehlt oder inkompatibel!
    echo    Fix: venv\Scripts\python -m spacy download de_core_news_md
    pause
    exit /b 1
)

REM Prüfe FastText
if not exist "lid.176.ftz" (
    echo ❌ FastText fehlt: lid.176.ftz
    pause
    exit /b 1
)

echo ✅ ALLE ABHÄNGIGKEITEN OK!

REM Altes Zeug löschen
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
if exist "varauto.spec" del varauto.spec

REM BAUE
echo.
echo 📦 Starte PyInstaller...
pyinstaller --onefile --name FloX --noconsole --icon=icon.ico ^
--add-data "lid.176.ftz;." ^
--collect-all spacy ^
--collect-all "de_core_news_md" ^
--hidden-import="spacy" ^
varAuto.py

REM Erfolg prüfen
if not exist "dist\varauto.exe" (
    echo ❌ FEHLER: EXE nicht erstellt!
    pause
    exit /b 1
)

echo.
echo ==================================================
echo ✅ BUILD ABGESCHLOSSEN!
echo ==================================================
echo 📁 EXE: %PROJECT_DIR%\dist\varauto.exe
echo.
echo ⚠️  TESTE JETZT:
echo    dist\varauto.exe --help
echo ==================================================
pause