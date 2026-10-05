@echo off
setlocal EnableExtensions
title Bonsai 2 27B (PQ2_0) - no speculative decoding - Tesla V100

REM Same service, but WITHOUT MTP speculative decoding.
REM Use this for very long context / long-document work: at ~100K depth the MTP draft
REM head also has to attend over the whole KV and measured ~10% SLOWER than plain decode.
REM For chat / short and medium prompts keep start-server.cmd (MTP on) instead.

set "BIN=C:\path\to\llama.cpp-bonsai\build-v100\bin"
set "MODEL=%~dp0Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf"

set "CTX=262144"
set "KVK=q8_0"
set "KVV=q8_0"
set "PORT=8080"
set "THREADS=16"

set CUDA_VISIBLE_DEVICES=0
set GGML_CUDA_DISABLE_GRAPHS=1

echo.
echo   model : %MODEL%
echo   ctx   : %CTX%   KV: %KVK% / %KVV%   MTP: OFF   port: %PORT%
echo.
ipconfig | findstr /R /C:"IPv4"
echo.

"%BIN%\llama-server.exe" ^
  -m "%MODEL%" ^
  -ngl 99 -fa on ^
  -c %CTX% -ctk %KVK% -ctv %KVV% ^
  -b 2048 -ub 512 -t %THREADS% ^
  -np 1 ^
  --host 0.0.0.0 --port %PORT% ^
  --jinja ^
  --log-file "%~dp0server.log"

echo.
echo   llama-server exited with code %errorlevel%
pause
