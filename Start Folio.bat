@echo off
setlocal
title Folio
cd /d "%~dp0"
set "FOLIO=%USERPROFILE%\Folio"
if exist "%FOLIO%\venv\Scripts\python.exe" goto run

echo Setting up Folio for the first time.
echo.
call :find_uv
if not defined UV call :install astral-sh.uv "uv, which installs Python and Folio's libraries"
if not defined UV call :find_uv
if not defined UV goto no_uv
call :find_ollama
if not defined OLLAMA call :install Ollama.Ollama "Ollama, which runs answer models on this PC. You can skip it if you'll only use Claude"
call :find_zotero
if not defined ZOTERO call :install Zotero.Zotero "Zotero, where your papers are kept"

echo.
echo Downloading Python and Folio's libraries, about 4 GB. This can take a while.
set "UV_PROJECT_ENVIRONMENT=%FOLIO%\venv"
set "UV_PYTHON_INSTALL_DIR=%FOLIO%\python"
set "UV_HTTP_TIMEOUT=1200"
call "%UV%" sync
if errorlevel 1 goto sync_failed

:run
"%FOLIO%\venv\Scripts\python.exe" -m folio %*
if errorlevel 1 pause
exit /b

:no_uv
echo.
echo Folio can't be set up without uv. Install it from https://docs.astral.sh/uv/ then start Folio again.
pause
exit /b 1

:sync_failed
echo.
echo Setup didn't finish. Check the internet connection, then start Folio again: it carries on where it stopped.
pause
exit /b 1

rem ------------------------------------------------------------------ helpers

:install
rem %1 is the winget package, %2 says what it's for.
echo Folio needs %~2.
call choice /c YN /m "Install it now"
if errorlevel 2 exit /b 0
call winget install --id %1 -e --accept-source-agreements --accept-package-agreements
if errorlevel 1 echo It couldn't be installed automatically. You can install %1 yourself later.
echo.
exit /b 0

:find_uv
set "UV="
for %%P in (uv.exe) do set "UV=%%~$PATH:P"
if not defined UV for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*") do if exist "%%D\uv.exe" set "UV=%%D\uv.exe"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
exit /b 0

:find_ollama
set "OLLAMA="
for %%P in (ollama.exe) do set "OLLAMA=%%~$PATH:P"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=1"
if not defined OLLAMA if exist "%FOLIO%\ollama\ollama.exe" set "OLLAMA=1"
exit /b 0

:find_zotero
set "ZOTERO="
if exist "%ProgramFiles%\Zotero\zotero.exe" set "ZOTERO=1"
if exist "%ProgramFiles(x86)%\Zotero\zotero.exe" set "ZOTERO=1"
if exist "%LOCALAPPDATA%\Zotero\zotero.exe" set "ZOTERO=1"
exit /b 0
