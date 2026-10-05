@echo off
setlocal EnableExtensions
title Bonsai 2 27B (PQ2_0 + MTP) - Tesla V100

REM =====================================================================
REM  Bonsai 2 27B ternary + MTP speculative decoding, LAN service
REM  Double-click to start. Stop with stop-server.cmd or by closing the window.
REM =====================================================================

set "BIN=C:\path\to\llama.cpp-bonsai\build-v100\bin"
set "MODEL=%~dp0Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf"

REM ---- tuning measured on this V100 (see README.md) ----
set "CTX=262144"
set "KVK=q8_0"
set "KVV=q8_0"
set "SPEC_NMAX=3"
set "PORT=8080"
set "THREADS=16"

set CUDA_VISIBLE_DEVICES=0
REM  with MTP, graphs=off is ~9% faster than on (per-step batch shapes differ)
set GGML_CUDA_DISABLE_GRAPHS=1
REM  CUDA state-path tuning from this project (all default-on, listed for clarity)
set GGML_CUDA_GDN_ROWS_READ=1
set GGML_CUDA_CONV_STATE_FUSION=1
set GGML_CUDA_FWHT_Q8_1=1
set GGML_V100_MMVQ_Q8_1=1

echo.
echo   model : %MODEL%
echo   ctx   : %CTX%   KV: %KVK% / %KVV%   MTP n-max: %SPEC_NMAX%   port: %PORT%
echo.
echo   LAN addresses (use one of these + :%PORT%) :
ipconfig | findstr /R /C:"IPv4"
echo.
echo   OpenAI-compatible API : http://<ip>:%PORT%/v1
echo   health check          : http://<ip>:%PORT%/health
echo.
echo   NOTE: a full %CTX%-token prompt takes a few minutes to prefill.
echo.

"%BIN%\llama-server.exe" ^
  -m "%MODEL%" ^
  -ngl 99 -fa on ^
  -c %CTX% -ctk %KVK% -ctv %KVV% ^
  -b 2048 -ub 512 -t %THREADS% ^
  -np 1 ^
  --host 0.0.0.0 --port %PORT% ^
  --jinja ^
  --spec-type draft-mtp --spec-draft-n-max %SPEC_NMAX% ^
  --log-file "%~dp0server.log"

echo.
echo   llama-server exited with code %errorlevel%
pause
