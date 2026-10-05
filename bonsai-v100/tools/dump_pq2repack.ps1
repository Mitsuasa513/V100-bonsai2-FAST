# Bit-exactness of the repacked PQ2_0 weight copy (GGML_CUDA_PQ2_0_REPACK) against the packed
# 34-byte layout, in both graph modes, short prompt + 32K prompt.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$p32k = if ($env:LLAMA_PROMPT_32K) { $env:LLAMA_PROMPT_32K } else { "C:\path\to\work\prompt_32k.txt" }
$out   = Join-Path $work "dump_pq2repack.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class PR {
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
Log "## PQ2_0 repack bit-exactness  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")
function Run-D($graphs, $repack, $tag, $promptArgs) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    Remove-Item Env:\GGML_CUDA_PQ2_0_REPACK -ErrorAction SilentlyContinue
    if (-not $repack) { $env:GGML_CUDA_PQ2_0_REPACK = '0' }
    $f = Join-Path $work "pr_$tag.bin"; if (Test-Path $f) { Remove-Item $f -Force }
    $env:GGML_V100_DUMP_LAST = $f
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 --temp 0 --seed 7 `
        --no-display-prompt -co off -st @promptArgs 1> (Join-Path $work "pr_$tag.txt") 2> (Join-Path $work "pr_$tag.err") | Out-Null
    $sw.Stop()
    Remove-Item Env:\GGML_V100_DUMP_LAST,Env:\GGML_CUDA_PQ2_0_REPACK,Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Log ("  ran {0,-20} wall={1,6:N1}s dump={2}" -f $tag, $sw.Elapsed.TotalSeconds, (Get-Item $f -ErrorAction SilentlyContinue).Length)
    return $f
}
$p_short = @('-c', '4096', '-p', 'The capital of France is', '-n', '32')
$off_a = Run-D $false $false 'gOFF_packed_a' $p_short
$on_a  = Run-D $false $true  'gOFF_repack_a' $p_short
$on_b  = Run-D $false $true  'gOFF_repack_b' $p_short
$gon   = Run-D $true  $true  'gON_repack'    $p_short
$gon_o = Run-D $true  $false 'gON_packed'    $p_short
Log ""
Log ("short: repack ON vs packed        : " + [PR]::Bits($off_a, $on_a))
Log ("short: repack repeatability       : " + [PR]::Bits($on_a, $on_b))
Log ("short: graphs ON (repack) vs gOFF : " + [PR]::Bits($on_a, $gon))
Log ("short: graphs ON packed vs gOFF   : " + [PR]::Bits($off_a, $gon_o))

if (Test-Path $p32k) {
    Log ""
    Log "32K prompt:"
    $p32 = @('-c', '40960', '-f', $p32k, '-n', '4')
    $k_o = Run-D $false $false 'gOFF_packed_32k' $p32
    $k_n = Run-D $false $true  'gOFF_repack_32k' $p32
    $k_g = Run-D $true  $true  'gON_repack_32k'  $p32
    Log ("32k: repack ON vs packed          : " + [PR]::Bits($k_o, $k_n))
    Log ("32k: graphs ON vs graphs OFF      : " + [PR]::Bits($k_n, $k_g))
}
Log ("log: $out")
