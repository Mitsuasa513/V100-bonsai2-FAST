# MTP A/B on the grafted Bonsai 2 27B PQ2_0 + MTP file:
#   baseline (same file, no --spec-type)  vs  draft-mtp at n-max 1/2/3
# Gates: greedy logits dump bit-identical, and the reported generation rate.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$work = if ($env:LLAMA_WORK_MTP) { $env:LLAMA_WORK_MTP } else { "C:\path\to\work\mtp" }
$model = Join-Path $work "Ternary-Bonsai-2-27B-PQ2_0-MTP.gguf"
$out   = Join-Path $work "ab_mtp.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class MTC {
    public static string Bits(string a, string b) {
        if (!File.Exists(a) || !File.Exists(b)) return "MISSING FILE";
        byte[] ba = File.ReadAllBytes(a), bb = File.ReadAllBytes(b);
        if (ba.Length != bb.Length) return string.Format("SIZE DIFFERS {0} vs {1}", ba.Length, bb.Length);
        long nBits = 0; double maxAbs = 0; long first = -1;
        for (int i = 0; i < ba.Length; i += 4) {
            int ia = BitConverter.ToInt32(ba, i), ib = BitConverter.ToInt32(bb, i);
            if (ia != ib) { nBits++; double d = Math.Abs((double)BitConverter.ToSingle(ba, i) - (double)BitConverter.ToSingle(bb, i)); if (d > maxAbs) maxAbs = d; if (first < 0) first = i / 4; }
        }
        if (nBits == 0) return string.Format("BIT-IDENTICAL ({0} logits)", ba.Length / 4);
        return string.Format("bitDiff={0}/{1} maxAbsDiff={2:E3} firstAt={3}", nBits, ba.Length / 4, maxAbs, first);
    }
}
"@
function Rate($outFile) {
    $m = Select-String -Path $outFile -Pattern 'Generation:\s*([0-9.]+)\s*t/s' | Select-Object -Last 1
    if (-not $m) { return 0.0 }
    if ($m.Line -match 'Generation:\s*([0-9.]+)') { return [double]$Matches[1] }
    return 0.0
}
function TextHash($path) {
    $lines = (Get-Content -LiteralPath $path) | Where-Object { $_ -notmatch '^\[ Prompt:' -and $_ -notmatch '^llama_' -and $_ -notmatch '^\[' }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    (($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n")))) | ForEach-Object { $_.ToString('x2') }) -join ''
}
function Run-One($tag, $specArgs, $n, $dump) {
    $f = Join-Path $work "mtp_$tag.bin"
    if ($dump) { if (Test-Path $f) { Remove-Item $f -Force }; $env:GGML_V100_DUMP_LAST = $f }
    else { Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 8192 `
        -p "The capital of France is" -n $n --temp 0 --seed 7 --no-display-prompt -co off -st `
        @specArgs 1> (Join-Path $work "mtp_$tag.out") 2> (Join-Path $work "mtp_$tag.err") | Out-Null
    $sw.Stop()
    Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
    $r = Rate (Join-Path $work "mtp_$tag.out")
    Log ("  {0,-16} gen={1,7:N2} t/s   wall={2,6:N1}s   text={3}" -f $tag, $r, $sw.Elapsed.TotalSeconds, (TextHash (Join-Path $work "mtp_$tag.out")).Substring(0,16))
    return @{ rate = $r; dump = $f; text = (TextHash (Join-Path $work "mtp_$tag.out")) }
}
Log "## MTP A/B  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("model: $model")
Log ("dll  : $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")
Log ""
Log "A) grafted trunk == original trunk? (logits dump, greedy, 32 tokens, no spec)"
$orig = Join-Path $work "trunk_orig.bin"
if (Test-Path $orig) { Remove-Item $orig -Force }
$env:GGML_V100_DUMP_LAST = $orig
& (Join-Path $bin 'llama-cli.exe') -m $env:BONSAI_MODEL `
    -ngl 99 -fa 1 -c 8192 -p "The capital of France is" -n 32 --temp 0 --seed 7 --no-display-prompt -co off -st `
    1> (Join-Path $work "trunk_orig.out") 2> (Join-Path $work "trunk_orig.err") | Out-Null
Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
$g = Run-One 'nospec_dump' @() 32 $true
Log ("  original vs grafted (no spec): " + [MTC]::Bits($orig, $g.dump))
Log ""
Log "B) correctness: greedy text must be identical with and without MTP (64 tokens)"
$t0 = Run-One 'nospec_t' @() 64 $false
$t1 = Run-One 'nmax1_t'  @('--spec-type','draft-mtp','--spec-draft-n-max','1') 64 $false
$t2 = Run-One 'nmax2_t'  @('--spec-type','draft-mtp','--spec-draft-n-max','2') 64 $false
$t3 = Run-One 'nmax3_t'  @('--spec-type','draft-mtp','--spec-draft-n-max','3') 64 $false
Log ("  nospec vs nmax1 : " + $(if ($t0.text -eq $t1.text) { 'TEXT IDENTICAL' } else { 'TEXT DIFFERS' }))
Log ("  nospec vs nmax2 : " + $(if ($t0.text -eq $t2.text) { 'TEXT IDENTICAL' } else { 'TEXT DIFFERS' }))
Log ("  nospec vs nmax3 : " + $(if ($t0.text -eq $t3.text) { 'TEXT IDENTICAL' } else { 'TEXT DIFFERS' }))
Log ""
Log "C) throughput (same prompt, 256 tokens, greedy)"
$b2 = Run-One 'nospec_256' @() 256 $false
foreach ($nm in 1,2,3,4) {
    Run-One ("nmax${nm}_256") @('--spec-type','draft-mtp','--spec-draft-n-max',"$nm") 256 $false
}
Log ("  gain nmax2 vs nospec = {0:+0.0;-0.0}%" -f (100.0*($t2.rate-$b2.rate)/[Math]::Max($b2.rate,0.001)))
Log ("log: $out")
