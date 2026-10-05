# Syntax/type check a single translation unit of the b11004-bonsai CUDA build by
# replaying the exact nvcc command line from compile_commands.json, but writing the
# object file into a temp dir (so the real build products are left untouched).
param([string]$Filter = "gated_delta_net.cu")

$ErrorActionPreference = 'Continue'

$tree = if ($env:LLAMA_TREE) { $env:LLAMA_TREE } else { "C:\path\to\llama.cpp-bonsai" }
$build = Join-Path $tree "build-v100"
$json  = Get-Content (Join-Path $build "compile_commands.json") -Raw | ConvertFrom-Json
$entry = $json | Where-Object { $_.file -like "*$Filter" } | Select-Object -First 1

if (-not $entry) {
    Write-Output "no compile_commands entry for $Filter"
    exit 1
}

$tmp = Join-Path $env:TEMP "cccheck"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$cmd = $entry.command
$cmd = $cmd -replace '-o\s+\S+\.obj', "-o `"$tmp\out.obj`""
$cmd = $cmd -replace '-Xcompiler=-Fd\S+', "-Xcompiler=-Fd$tmp\"

$bat = Join-Path $tmp "cc.bat"
@(
    "@echo off",
    'call $env:VCVARS64 >nul 2>&1',
    "cd /d `"$($entry.directory)`"",
    $cmd,
    "echo CC_EXIT=%errorlevel%"
) | Set-Content -Path $bat -Encoding ASCII

Write-Output "checking $($entry.file)"
$raw = Join-Path $tmp "raw.log"
cmd /c $bat > $raw 2>&1
$rc = Select-String -Path $raw -Pattern "CC_EXIT=(\d+)" | Select-Object -Last 1
Write-Output "obj: $((Get-Item (Join-Path $tmp 'out.obj') -ErrorAction SilentlyContinue).Length) bytes, mtime $((Get-Item (Join-Path $tmp 'out.obj') -ErrorAction SilentlyContinue).LastWriteTime)"
Write-Output "raw log: $((Get-Item $raw).Length) bytes"
if ($rc) { Write-Output $rc.Line.Trim() }
Select-String -Path $raw -Pattern ": error|: fatal|error #|catastrophic|CC_EXIT" | Select-Object -First 60 | ForEach-Object { $_.Line }
Write-Output "done"
