# MTP at depth: no-spec vs draft-mtp (n-max 4) with a 32K and a ~96K prompt, interleaved rounds.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$work = if ($env:LLAMA_WORK_MTP) { $env:LLAMA_WORK_MTP } else { "C:\path\to\work\mtp" }
$model = Join-Path $work "Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf"
$p32k = if ($env:LLAMA_PROMPT_32K) { $env:LLAMA_PROMPT_32K } else { "C:\path\to\work\prompt_32k.txt" }
$p96k  = Join-Path $work "prompt_96k.txt"
$out   = Join-Path $work "bench_mtp_depth.txt"
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
function TextHash($path) {
    $lines = (Get-Content -LiteralPath $path) | Where-Object { $_ -notmatch '^\[ Prompt:' -and $_ -notmatch '^llama_' -and $_ -notmatch '^\[' }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    (($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n")))) | ForEach-Object { $_.ToString('x2') }) -join ''
}
function Run($nmax, $prompt, $ctx, $n, $tag) {
    $o = Join-Path $work "d_${tag}_$nmax.out"
    $a = @('-m', $model, '-ngl', '99', '-fa', '1', '-c', "$ctx", '-f', $prompt, '-n', "$n",
           '--temp', '0', '--seed', '7', '--no-display-prompt', '-co', 'off', '-st')
    if ($nmax -gt 0) { $a += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$nmax") }
    & (Join-Path $bin 'llama-cli.exe') @a 1> $o 2> (Join-Path $work "d_${tag}_$nmax.err") | Out-Null
    return @{ rate = (Rate $o); text = (TextHash $o) }
}

# ~96K prompt = the 32K prompt three times
if (-not (Test-Path $p96k)) {
    $t = Get-Content $p32k -Raw
    ($t + $t + $t) | Out-File $p96k -Encoding utf8
    Log ("built $p96k ({0:N2} MB)" -f ((Get-Item $p96k).Length/1MB))
}

Log "## MTP at depth  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
foreach ($cfg in @(
        @{ name='32K'; prompt=$p32k; ctx=40960; n=128; rounds=3 },
        @{ name='96K'; prompt=$p96k; ctx=114688; n=128; rounds=2 })) {
    Log ""
    Log ("--- {0} (n={1}, rounds={2}) ---" -f $cfg.name, $cfg.n, $cfg.rounds)
    $base = @(); $mtp = @(); $bt = ''; $mt = ''
    for ($r = 1; $r -le $cfg.rounds; $r++) {
        $a = Run 0 $cfg.prompt $cfg.ctx $cfg.n $cfg.name
        $b = Run 4 $cfg.prompt $cfg.ctx $cfg.n $cfg.name
        $base += $a.rate; $mtp += $b.rate
        if ($r -eq 1) { $bt = $a.text; $mt = $b.text }
        Log ("  round {0}: nospec={1:N1}  nmax4={2:N1}" -f $r, $a.rate, $b.rate)
    }
    $ab = ($base | Measure-Object -Maximum).Maximum
    $am = ($base | Sort-Object)[[int][math]::Floor($base.Count/2)]
    $bb = ($mtp | Measure-Object -Maximum).Maximum
    $bm = ($mtp | Sort-Object)[[int][math]::Floor($mtp.Count/2)]
    Log ("  best   nospec={0:N2}  nmax4={1:N2}  = {2:+0.0;-0.0}%" -f $ab, $bb, (100.0*($bb-$ab)/[Math]::Max($ab,0.001)))
    Log ("  median nospec={0:N2}  nmax4={1:N2}  = {2:+0.0;-0.0}%" -f $am, $bm, (100.0*($bm-$am)/[Math]::Max($am,0.001)))
    Log ("  greedy text nospec vs nmax4: " + $(if ($bt -eq $mt) { 'IDENTICAL' } else { 'DIFFERS' }))
}
Log ("log: $out")
