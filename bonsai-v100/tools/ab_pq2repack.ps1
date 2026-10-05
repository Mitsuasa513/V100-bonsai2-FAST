# Interleaved A/B of the repacked PQ2_0 weight copy (GGML_CUDA_PQ2_0_REPACK, default on),
# graphs OFF. Alternating rounds so machine drift hits both arms equally.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "ab_pq2repack.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Log "## PQ2_0 repack A/B  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
function Bench($on, $d, $rounds, $nGen) {
    if ($on) { Remove-Item Env:\GGML_CUDA_PQ2_0_REPACK -ErrorAction SilentlyContinue }
    else     { $env:GGML_CUDA_PQ2_0_REPACK = '0' }
    $txt = & (Join-Path $bin 'llama-bench.exe') -m $model -ngl 99 -fa 1 -p 512 -n $nGen -d $d -r $rounds -o csv 2>&1 |
        Out-String
    Remove-Item Env:\GGML_CUDA_PQ2_0_REPACK -ErrorAction SilentlyContinue
    $res = @{ pp = 0.0; tg = 0.0 }
    foreach ($line in ($txt -split "`r?`n")) {
        if ($line -notmatch '^"unknown"') { continue }
        $f = $line.Trim('"') -split '","'
        $nPrompt = $f[$f.Count - 8]; $nGenF = $f[$f.Count - 7]
        $avgTs = [double]$f[$f.Count - 2]
        if ($nPrompt -eq '512' -and $nGenF -eq '0') { $res.pp = $avgTs }
        if ($nPrompt -eq '0'   -and $nGenF -ne '0') { $res.tg = $avgTs }
    }
    return $res
}
function Stats($xs) { return @{ best = ($xs | Measure-Object -Maximum).Maximum; med = ($xs | Sort-Object)[[int]([math]::Floor($xs.Count/2))] } }
foreach ($d in @(0, 32768, 90112)) {
    $rounds = if ($d -eq 0) { 6 } else { 3 }
    $nGen   = if ($d -eq 0) { 256 } else { 128 }
    $on = @(); $off = @()
    for ($k = 1; $k -le $rounds; $k++) {
        $off += ,(Bench $false $d 3 $nGen)
        $on  += ,(Bench $true  $d 3 $nGen)
    }
    $onTg  = $on  | ForEach-Object { $_.tg }
    $offTg = $off | ForEach-Object { $_.tg }
    $onPp  = $on  | ForEach-Object { $_.pp }
    $offPp = $off | ForEach-Object { $_.pp }
    Log ""
    Log "--- d=$d  (tg n=$nGen, r=3 x$rounds) ---"
    Log ("  repack ON : " + (($onTg  | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    Log ("  packed OFF: " + (($offTg | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    Log ("  pp512 on/off: " + (($onPp | ForEach-Object { '{0:N1}' -f $_ }) -join '  ') + "   /   " + (($offPp | ForEach-Object { '{0:N1}' -f $_ }) -join '  '))
    $onS = Stats $onTg; $offS = Stats $offTg
    Log ("  tg best {0:N2} vs {1:N2} = {2:+0.00;-0.00}%   median {3:N2} vs {4:N2} = {5:+0.00;-0.00}%" -f `
        $onS.best, $offS.best, (100.0*($onS.best-$offS.best)/$offS.best), $onS.med, $offS.med, (100.0*($onS.med-$offS.med)/$offS.med))
    $onPb = ($onPp | Measure-Object -Maximum).Maximum; $offPb = ($offPp | Measure-Object -Maximum).Maximum
    Log ("  pp512 best {0:N1} vs {1:N1} = {2:+0.00;-0.00}%" -f $onPb, $offPb, (100.0*($onPb-$offPb)/$offPb))
}
Log ("log: $out")
