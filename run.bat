@echo off

REM Run main FastAPI app in new window
start cmd /k "python main.py"

REM Run chat API in another window
start cmd /k "python chat_api.py"

REM Run static frontend server
python -m http.server 3000

pause
