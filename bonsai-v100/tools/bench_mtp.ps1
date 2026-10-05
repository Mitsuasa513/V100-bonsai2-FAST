# Interleaved MTP sweep: n-max 0 (no spec) / 1 / 2 / 3 / 4 / 6, greedy, same prompt.
# Alternating rounds so machine drift hits every setting equally; reports best and median.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$work = if ($env:LLAMA_WORK_MTP) { $env:LLAMA_WORK_MTP } else { "C:\path\to\work\mtp" }
$model = Join-Path $work "Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf"
$p32k = if ($env:LLAMA_PROMPT_32K) { $env:LLAMA_PROMPT_32K } else { "C:\path\to\work\prompt_32k.txt" }
$out   = Join-Path $work "bench_mtp.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Set-Location $bin
Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
function Rate($outFile) {
    $m = Select-String -Path $outFile -Pattern 'Generation:\s*([0-9.]+)\s*t/s' | Select-Object -Last 1
    if ($m -and $m.Line -match 'Generation:\s*([0-9.]+)') { return [double]$Matches[1] }
    return 0.0
}
function Run($nmax, $n, $depth) {
    # llama-cli has no -d; depth comes from feeding the 32K prompt file with a large context
    $promptArgs = if ($depth -gt 0) { @('-c', '40960', '-f', $p32k, '-n', "$n") } else { @('-c', '8192', '-n', "$n") }
    $specArgs = if ($nmax -gt 0) { @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$nmax") } else { @() }
    $tag = "nm$nmax"
    $o = Join-Path $work "b_${tag}_$n.out"
    $a = @('-m', $model, '-ngl', '99', '-fa', '1', '--temp', '0', '--seed', '7', '--no-display-prompt', '-co', 'off', '-st') + $specArgs
    if ($depth -eq 0) { $a += @('-p', 'The capital of France is') }
    $a += $promptArgs
    & (Join-Path $bin 'llama-cli.exe') @a 1> $o 2> (Join-Path $work "b_${tag}_$n.err") | Out-Null
    return (Rate $o)
}
$settings = @(0, 1, 2, 3, 4, 6)
Log "## MTP n-max sweep  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
foreach ($cfg in @(@{name='d=0'; n=512; depth=0; rounds=4}, @{name='32K'; n=256; depth=32768; rounds=2})) {
    Log ""
    Log ("--- {0}  (n={1}, rounds={2}, greedy, same prompt) ---" -f $cfg.name, $cfg.n, $cfg.rounds)
    $res = @{}
    foreach ($s in $settings) { $res[$s] = @() }
    for ($r = 1; $r -le $cfg.rounds; $r++) {
        foreach ($s in $settings) {
            $v = Run $s $cfg.n $cfg.depth
            $res[$s] += $v
        }
        Log ("  round {0}: " + (($settings | ForEach-Object { "nm{0}={1:N1}" -f $_, ($res[$_])[-1] }) -join '  '))
    }
    Log ""
    Log ("  {0,-14} {1,10} {2,10} {3,10} {4,10}" -f 'setting', 'best', 'median', 'gain-best', 'gain-med')
    $base = $res[0]
    $bb = ($base | Measure-Object -Maximum).Maximum
    $bm = ($base | Sort-Object)[[int][math]::Floor($base.Count/2)]
    foreach ($s in $settings) {
        $v = $res[$s]
        $best = ($v | Measure-Object -Maximum).Maximum
        $med  = ($v | Sort-Object)[[int][math]::Floor($v.Count/2)]
        Log ("  {0,-14} {1,10:N2} {2,10:N2} {3,10} {4,10}" -f `
            ("nm$s" + $(if ($s -eq 0) { ' (nospec)' } else { '' })), $best, $med, (100.0*($best-$bb)/[Math]::Max($bb,0.001)), (100.0*($med-$bm)/[Math]::Max($bm,0.001)))
    }
}
Log ("log: $out")
