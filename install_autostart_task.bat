@echo off
title Install Auto-Start Task
chcp 1251 >nul 2>&1

echo ================================================
echo   Dark Messenger - Auto-Start Installation
echo ================================================
echo.

:: Запуск от администратора
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting Administrator privileges...
    powershell start -verb runas '%0'
    exit /b
)

:: Путь к вашему скрипту (ИЗМЕНИТЕ НА ВАШ)
set "SCRIPT_PATH=E:\BlockcoinWitres\start_app.bat"

:: Проверка существования
if not exist "%SCRIPT_PATH%" (
    echo ERROR: Script not found: %SCRIPT_PATH%
    echo Please edit the SCRIPT_PATH variable in this file
    pause
    exit /b 1
)

echo Installing Dark Messenger auto-start task...
echo Script: %SCRIPT_PATH%
echo.

:: Создаем задачу с отложенным запуском
schtasks /create /tn "DarkMessenger" /tr "\"%SCRIPT_PATH%\"" /sc onlogon /delay 0001:00 /f

if %errorlevel%==0 (
    echo.
    echo ================================================
    echo   ✓ Auto-start task installed successfully!
    echo ================================================
    echo.
    echo   Task name: DarkMessenger
    echo   Trigger: At user logon
    echo   Delay: 1 minute
    echo   Script: %SCRIPT_PATH%
    echo.
    echo To remove: schtasks /delete /tn "DarkMessenger" /f
) else (
    echo.
    echo ================================================
    echo   ✗ Failed to install task
    echo ================================================
    echo.
    echo Possible issues:
    echo   1. Make sure you're running as Administrator
    echo   2. Check Task Scheduler service is running
)

echo.
pause