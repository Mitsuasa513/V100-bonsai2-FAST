# One-shot acceptance suite for the V100 CUDA state-path fusions.
#
# Why logits and not text: the first version of the conv-state fusion folded the graph's
# write-back CPY into the same kernel that reads the cache rows and was racy. Generated
# text and perplexity were byte-identical for it -- only a bit-exact comparison of the
# dumped logits exposed the run-to-run drift (up to 2e-1). Text is still checked, but the
# gate is the logits dump now.
#
# Stages
#   1. build freshness (DLL newer than sources, no error lines in build-log.txt)
#   2. logits bit-exactness, short greedy prompt, 4 configurations x 2 graph modes
#   3. logits bit-exactness, 32K prompt (exercises the long-token prefill path)
#   4. generated text byte-compare, both graph modes
#   5. llama-perplexity per-chunk values
#   6. multi-sequence (llama-server -np 2, delegated to multiseq2.ps1)
#   7. perf: tg128 at d=0 and d=90112, pp512/pp8192
#
# Switches under test: GGML_CUDA_GDN_ROWS_READ=0 (A/B reference), GGML_CUDA_CONV_STATE_FUSION=0

$ErrorActionPreference = 'Continue'
$tree = if ($env:LLAMA_TREE) { $env:LLAMA_TREE } else { "C:\path\to\llama.cpp-bonsai" }
$bin   = Join-Path $tree "build-v100\bin"
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$p32k = if ($env:LLAMA_PROMPT_32K) { $env:LLAMA_PROMPT_32K } else { "C:\path\to\work\prompt_32k.txt" }
$out   = Join-Path $work "acceptance.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin
$results = [ordered]@{}
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
function Record($key, $ok, $detail) {
    $results[$key] = $ok
    Log ("  [{0}] {1,-26} {2}" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $key, $detail)
}

Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class ACC {
    public static string Bits(string a, string b) {
        if (!File.Exists(a) || !File.Exists(b)) return "MISSING FILE";
        byte[] ba = File.ReadAllBytes(a), bb = File.ReadAllBytes(b);
        if (ba.Length != bb.Length) return string.Format("SIZE DIFFERS {0} vs {1}", ba.Length, bb.Length);
        int n = ba.Length / 4;
        float[] fa = new float[n], fb = new float[n];
        Buffer.BlockCopy(ba, 0, fa, 0, ba.Length); Buffer.BlockCopy(bb, 0, fb, 0, bb.Length);
        long nBits = 0; double maxAbs = 0; long first = -1; float fa0 = 0, fb0 = 0;
        for (int i = 0; i < n; i++) {
            if (BitConverter.ToInt32(ba, 4 * i) != BitConverter.ToInt32(bb, 4 * i)) {
                nBits++;
                double d = Math.Abs((double)fa[i] - (double)fb[i]);
                if (d > maxAbs) maxAbs = d;
                if (first < 0) { first = i; fa0 = fa[i]; fb0 = fb[i]; }
            }
        }
        if (nBits == 0) return string.Format("BIT-IDENTICAL ({0} logits)", n);
        return string.Format("bitDiff={0}/{1} maxAbsDiff={2:E3} firstAt={3} ({4} vs {5})", nBits, n, maxAbs, first, fa0, fb0);
    }
    public static bool Same(string a, string b) { return Bits(a, b).StartsWith("BIT-IDENTICAL"); }
}
"@

"" | Out-File $out -Encoding utf8
Log "## acceptance suite  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ------------------------------------------------------------------ 1. build freshness
Log ""
Log "1) build freshness"
$dll = Get-Item (Join-Path $bin 'ggml-cuda.dll')
$srcNewest = Get-ChildItem (Join-Path $tree 'ggml\src\ggml-cuda') -Include *.cu,*.cuh -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$errLines = @(Select-String -Path (Join-Path $tree 'build-v100\build-log.txt') -Pattern ": error|FAILED" |
    Where-Object { $_.Line -notmatch 'UI:|HF download' })
Log ("  dll=$($dll.LastWriteTime)  newest source=$($srcNewest.Name) $($srcNewest.LastWriteTime)")
Record 'build.fresh' ($dll.LastWriteTime -gt $srcNewest.LastWriteTime) "dll newer than sources"
Record 'build.no_errors' ($errLines.Count -eq 0) "error lines in build-log.txt: $($errLines.Count)"

function Set-Cfg($graphs, $mode) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ,Env:\GGML_CUDA_CONV_STATE_FUSION -ErrorAction SilentlyContinue
    if ($mode -eq 'gather')   { $env:GGML_CUDA_GDN_ROWS_READ = '0' }
    if ($mode -eq 'conv-off') { $env:GGML_CUDA_CONV_STATE_FUSION = '0' }
}

# runs llama-cli with the logits dump hook enabled, returns the dump path
function Dump($graphs, $mode, $tag, $promptArgs, $n) {
    Set-Cfg $graphs $mode
    $f = Join-Path $work "acc_dump_$tag.bin"
    $t = Join-Path $work "acc_dump_$tag.txt"
    foreach ($p in @($f, $t)) { if (Test-Path $p) { Remove-Item $p -Force } }
    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 --temp 0 --seed 7 `
        --no-display-prompt -co off -st -n $n @promptArgs 1> $t 2> (Join-Path $work "acc_dump_$tag.err") | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
    if (-not (Test-Path $f)) { return $f }
    return $f
}

# ------------------------------------------------- 2. logits, short prompt, 8 configs
Log ""
Log "2) logits bit-exactness (short prompt, greedy, 32 tokens)"
$ref = Dump $false 'gather' 'gOFF_gather' @('-c', '4096', '-p', 'The capital of France is') 32
$short = [ordered]@{}
foreach ($g in @($false, $true)) {
    $gt = if ($g) { 'gON' } else { 'gOFF' }
    foreach ($mode in @('default', 'gather', 'conv-off')) {
        $tag = "${gt}_${mode}_b"
        $f = Dump $g $mode $tag @('-c', '4096', '-p', 'The capital of France is') 32
        $short[$tag] = $f
        if ($tag -eq 'gOFF_gather_b') { continue }
        $same = [ACC]::Same($ref, $f)
        Record "logits.$tag" $same ([ACC]::Bits($ref, $f))
    }
}
# run the reference again as a repeatability control
$ref2 = Dump $false 'gather' 'gOFF_gather_r2' @('-c', '4096', '-p', 'The capital of France is') 32
Record 'logits.repeatability' ([ACC]::Same($ref, $ref2)) ([ACC]::Bits($ref, $ref2))

# ------------------------------------------------------------- 3. logits, 32K prompt
Log ""
Log "2b) logits bit-exactness: ssm_alpha/ssm_beta matvec pair (delegated to dump_pair.ps1)"
$dpOut = Join-Path $work "dump_pair.txt"
if (Test-Path $dpOut) { Remove-Item $dpOut }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $work 'dump_pair.ps1') 2>&1 | Out-Null
$dp = if (Test-Path $dpOut) { Get-Content $dpOut -Raw } else { '' }
$dpChecks = @(($dp -split "`n") | Where-Object { $_ -match 'BIT-IDENTICAL|bitDiff=|SIZE DIFFERS|MISSING' })
$dpOk = ($dp -notmatch 'bitDiff=|SIZE DIFFERS|MISSING') -and ($dp -match 'BIT-IDENTICAL')
Record 'logits.pair' $dpOk (($dpChecks | ForEach-Object { $_.Trim() }) -join ' ; ')

if (Test-Path $p32k) {
    Log ""
    Log "3) logits bit-exactness (32K prompt -> long-token prefill path)"
    $r32 = Dump $false 'gather' 'gOFF_gather_32k' @('-c', '40960', '-f', $p32k) 4
    foreach ($g in @($false, $true)) {
        $gt = if ($g) { 'gON' } else { 'gOFF' }
        $f = Dump $g 'default' "${gt}_default_32k" @('-c', '40960', '-f', $p32k) 4
        Record "logits32k.$gt" ([ACC]::Same($r32, $f)) ([ACC]::Bits($r32, $f))
    }
} else {
    Log ""
    Log "3) skipped (prompt_32k.txt not found)"
}

# ------------------------------------------------------------------- 4. text compare
Log ""
Log "4) generated text byte-compare (default vs full gather)"
function NormHash($path) {
    $lines = (Get-Content -LiteralPath $path) | Where-Object { $_ -notmatch '^\[ Prompt:' -and $_ -notmatch '^llama_' }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    (($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n")))) |
        ForEach-Object { $_.ToString('x2') }) -join ''
}
foreach ($g in @($true, $false)) {
    $gt = if ($g) { 'gON' } else { 'gOFF' }
    $h = @{}
    foreach ($mode in @('default', 'gather')) {
        Set-Cfg $g $mode
        $t = Join-Path $work "acc_text_${gt}_$mode.txt"
        & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 4096 `
            -p "The capital of France is" -n 192 --temp 0 --seed 7 --no-display-prompt -co off -st `
            1> $t 2> (Join-Path $work "acc_text_${gt}_$mode.err") | Out-Null
        $h[$mode] = NormHash $t
    }
    Record "text.$gt" ($h['default'] -eq $h['gather']) "sha=$($h['default'].Substring(0,16))"
}

# -------------------------------------------------------------------- 5. perplexity
Log ""
Log "5) perplexity (1536 tokens, 3 chunks)"
$ppl = @{}
foreach ($mode in @('default', 'gather')) {
    Set-Cfg $false $mode
    $o = Join-Path $work "acc_ppl_$mode.txt"
    & (Join-Path $bin 'llama-perplexity.exe') -m $model -ngl 99 -fa 1 -c 512 -t 4 -f (Join-Path $work 'ppl_text_big.txt') 1> $o 2>&1 | Out-Null
    $ppl[$mode] = ((Get-Content $o | Select-String -Pattern '^\[' | ForEach-Object { $_.Line.Trim() }) -join '')
}
Record 'ppl' ($ppl['default'] -eq $ppl['gather']) $ppl['default']

# ---------------------------------------------------------------- 6. multi sequence
Log ""
Log "6) multi-sequence (llama-server -np 2) -- delegated to multiseq2.ps1"
$msOut = Join-Path $work "multiseq2_results.txt"
if (Test-Path $msOut) { Remove-Item $msOut }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $work 'multiseq2.ps1') 2>&1 | Out-Null
$ms = if (Test-Path $msOut) { Get-Content $msOut -Raw } else { '' }
$msOk = ($ms -notmatch 'DIFFERS') -and ($ms -match 'IDENTICAL')
Record 'multiseq' $msOk ((($ms -split "`n") | Where-Object { $_ -match 'IDENTICAL|DIFFERS' }) -join ' ; ')

# ---------------------------------------------------------------------------- 7. perf
Log ""
Log "7) perf (graphs OFF, -fa 1, -ngl 99)"
$perf = @{}
foreach ($mode in @('default', 'gather')) {
    Set-Cfg $false $mode
    foreach ($d in @(0, 90112)) {
        $r = & (Join-Path $bin 'llama-bench.exe') -m $model -ngl 99 -fa 1 -p 512 -n 128 -d $d -r 3 2>&1 |
            Select-String -Pattern 'pp512|tg128' | ForEach-Object { $_.Line.Trim() }
        $perf["$mode/$d"] = $r
        foreach ($line in $r) { Log "  [$mode d=$d] $line" }
    }
}
Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ,Env:\GGML_CUDA_CONV_STATE_FUSION,Env:\GGML_CUDA_DISABLE_GRAPHS,Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
# The +/- column of llama-bench is UTF-8; depending on the console code page the sign can arrive as
# anything, so read the column that FOLLOWS the test name instead of matching the sign character.
function TgOf($lines) {
    $m = ($lines | Select-String -Pattern 'tg128' | Select-Object -First 1)
    if (-not $m) { return 0.0 }
    $f = $m.Line -split '\|'
    for ($i = 0; $i -lt $f.Count - 1; $i++) {
        if ($f[$i].Trim() -match '^tg128') {
            if ($f[$i + 1].Trim() -match '^([0-9]+\.[0-9]+)') { return [double] $Matches[1] }
        }
    }
    return 0.0
}
foreach ($d in @(0, 90112)) {
    $a = TgOf $perf["default/$d"]; $b = TgOf $perf["gather/$d"]
    $pct = if ($b -gt 0) { 100.0 * ($a - $b) / $b } else { 0.0 }
    Record "perf.d$d" ($pct -gt 0.0) ("tg128 {0:N2} vs {1:N2} = {2:+0.00;-0.00}%" -f $a, $b, $pct)
}

Log "  -- matvec pair on/off at d=0 --"
$pair = @{}
foreach ($mode in @('pair-on', 'pair-off')) {
    Set-Cfg $false 'default'
    if ($mode -eq 'pair-off') { $env:GGML_CUDA_MMVF_PAIR = '0' }
    $r = & (Join-Path $bin 'llama-bench.exe') -m $model -ngl 99 -fa 1 -p 512 -n 128 -d 0 -r 3 2>&1 |
        Select-String -Pattern 'pp512|tg128' | ForEach-Object { $_.Line.Trim() }
    $pair[$mode] = $r
    foreach ($line in $r) { Log "  [$mode] $line" }
}
Remove-Item Env:\GGML_CUDA_MMVF_PAIR -ErrorAction SilentlyContinue
$pa = TgOf $pair['pair-on']; $pb = TgOf $pair['pair-off']
$ppct = if ($pb -gt 0) { 100.0 * ($pa - $pb) / $pb } else { 0.0 }
Record 'perf.pair.d0' ($ppct -gt 0.0) ("tg128 {0:N2} vs {1:N2} = {2:+0.00;-0.00}%" -f $pa, $pb, $ppct)

# ------------------------------------------------------------------------- summary
Log ""
$failed = @($results.GetEnumerator() | Where-Object { -not $_.Value })
Log ("SUMMARY: {0}/{1} checks passed" -f ($results.Count - $failed.Count), $results.Count)
if ($failed.Count -gt 0) { Log ("FAILED: " + (($failed | ForEach-Object { $_.Key }) -join ', ')) }
Log ("log: $out")
if ($failed.Count -gt 0) { exit 1 }
