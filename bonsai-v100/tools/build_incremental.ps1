# Incremental build of the b11004-bonsai tree, same steps as
# p1\build_bonsai_errors_only.bat (vcvars64 + ninja llama-cli llama-bench),
# logging to the same build-v100\build-log.txt.

$ErrorActionPreference = 'Continue'

$tree = if ($env:LLAMA_TREE) { $env:LLAMA_TREE } else { "C:\path\to\llama.cpp-bonsai" }
$build = Join-Path $tree "build-v100"
$log   = Join-Path $build "build-log.txt"
$vc = if ($env:VCVARS64) { $env:VCVARS64 } else { "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" }

$bat = Join-Path $env:TEMP "cccheck\build.bat"
New-Item -ItemType Directory -Force -Path (Split-Path $bat) | Out-Null

@(
    "@echo off",
    "call `"$vc`" >nul 2>&1",
    "if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )",
    "cd /d `"$build`"",
    "ninja -j 16 llama-cli llama-bench llama-perplexity llama-server > `"$log`" 2>&1",
    "echo NINJA_RC=%errorlevel%"
) | Set-Content -Path $bat -Encoding ASCII

$sw = [Diagnostics.Stopwatch]::StartNew()
cmd /c $bat 2>&1 | Select-Object -Last 3
$sw.Stop()
Write-Output ("build wall time: {0:n1} s" -f $sw.Elapsed.TotalSeconds)

Write-Output "---- error lines ----"
Select-String -Path $log -Pattern ": error|FAILED:|ninja: build stopped" | Select-Object -First 40 | ForEach-Object { $_.Line }
Write-Output "---- tail ----"
Get-Content $log -Tail 8
Get-Item (Join-Path $build "bin\llama-cli.exe"), (Join-Path $build "bin\llama-bench.exe") |
    Select-Object LastWriteTime, Length, Name | Format-Table -AutoSize
