@echo off
echo stopping llama-server ...
taskkill /IM llama-server.exe /F
timeout /t 2 >nul
echo done.
