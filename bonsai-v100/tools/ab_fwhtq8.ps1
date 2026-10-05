# Interleaved A/B of GGML_CUDA_FWHT_Q8_1 (default on) at d=0 / 32K / 90K, graphs off.
# Alternating rounds so a drifting machine hits both arms equally.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "ab_fwhtq8.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Log "## FWHT->q8_1 A/B  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# llama-bench -o csv: ... n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
# returns @{ pp = <avg_ts of pp512>; tg = <avg_ts of tg128> }
function Bench($on, $d, $rounds, $nGen) {
    if ($on) { Remove-Item Env:\GGML_CUDA_FWHT_Q8_1 -ErrorAction SilentlyContinue }
    else     { $env:GGML_CUDA_FWHT_Q8_1 = '0' }
    $txt = & (Join-Path $bin 'llama-bench.exe') -m $model -ngl 99 -fa 1 -p 512 -n $nGen -d $d -r $rounds -o csv 2>&1 |
        Out-String
    Remove-Item Env:\GGML_CUDA_FWHT_Q8_1 -ErrorAction SilentlyContinue
    $res = @{ pp = 0.0; tg = 0.0 }
    foreach ($line in ($txt -split "`r?`n")) {
        if ($line -notmatch '^"unknown"') { continue }
        $f = $line.Trim('"') -split '","'
        $nPrompt = $f[$f.Count - 8]; $nGen = $f[$f.Count - 7]
        $avgTs = [double]$f[$f.Count - 2]
        if ($nPrompt -eq '512' -and $nGen -eq '0') { $res.pp = $avgTs }
        if ($nPrompt -eq '0'   -and $nGen -ne '0') { $res.tg = $avgTs }
    }
    return $res
}
function Stats($xs) { return @{ best = ($xs | Measure-Object -Maximum).Maximum; med = ($xs | Sort-Object)[[int]([math]::Floor($xs.Count/2))] } }
foreach ($d in @(0, 32768, 90112)) {
    $rounds = if ($d -eq 0) { 8 } elseif ($d -eq 32768) { 4 } else { 3 }
    $nGen   = if ($d -eq 0) { 512 } elseif ($d -eq 32768) { 256 } else { 128 }
    $on = @(); $off = @()
    for ($k = 1; $k -le $rounds; $k++) {
        $on  += ,(Bench $true  $d 3 $nGen)
        $off += ,(Bench $false $d 3 $nGen)
    }
    $onTg  = $on  | ForEach-Object { $_.tg }
    $offTg = $off | ForEach-Object { $_.tg }
    $onPp  = $on  | ForEach-Object { $_.pp }
    $offPp = $off | ForEach-Object { $_.pp }
    Log ""
    Log "--- d=$d ---"
    Log ("  tg128 on : " + (($onTg  | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    Log ("  tg128 off: " + (($offTg | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    Log ("  pp512 on : " + (($onPp  | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    Log ("  pp512 off: " + (($offPp | ForEach-Object { '{0:N2}' -f $_ }) -join '  '))
    $onS  = Stats $onTg;  $offS  = Stats $offTg
    $onBest = $onS.best; $offBest = $offS.best; $onMed = $onS.med; $offMed = $offS.med
    Log ("  tg128 best {0:N2} vs {1:N2} = {2:+0.00;-0.00}%   median {3:N2} vs {4:N2} = {5:+0.00;-0.00}%" -f `
        $onBest, $offBest, (100.0*($onBest-$offBest)/$offBest), $onMed, $offMed, (100.0*($onMed-$offMed)/$offMed))
    $onPpBest  = ($onPp  | Measure-Object -Maximum).Maximum
    $offPpBest = ($offPp | Measure-Object -Maximum).Maximum
    Log ("  pp512 best {0:N2} vs {1:N2} = {2:+0.00;-0.00}%" -f $onPpBest, $offPpBest, (100.0*($onPpBest-$offPpBest)/$offPpBest))
}
Log ("log: $out")
