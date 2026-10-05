# Bit-exactness of the FWHT -> q8_1 pre-quantization (GGML_CUDA_FWHT_Q8_1) against the
# separate quantizer, in both graph modes, short prompt + 32K prompt.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$p32k = if ($env:LLAMA_PROMPT_32K) { $env:LLAMA_PROMPT_32K } else { "C:\path\to\work\prompt_32k.txt" }
$out   = Join-Path $work "dump_fwhtq8.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class FQ {
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
Log "## FWHT->q8_1 bit-exactness  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")
function Run-D($graphs, $fq, $tag, $promptArgs) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    Remove-Item Env:\GGML_CUDA_FWHT_Q8_1 -ErrorAction SilentlyContinue
    if (-not $fq) { $env:GGML_CUDA_FWHT_Q8_1 = '0' }
    $f = Join-Path $work "fq_$tag.bin"; if (Test-Path $f) { Remove-Item $f -Force }
    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 --temp 0 --seed 7 `
        --no-display-prompt -co off -st @promptArgs 1> (Join-Path $work "fq_$tag.txt") 2> (Join-Path $work "fq_$tag.err") | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST,Env:\GGML_CUDA_FWHT_Q8_1,Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Log ("  ran {0,-22} dump={1}" -f $tag, (Get-Item $f -ErrorAction SilentlyContinue).Length)
    return $f
}
$p_short = @('-c', '4096', '-p', 'The capital of France is', '-n', '32')
$off_a = Run-D $false $false 'gOFF_off_a' $p_short
$off_b = Run-D $false $false 'gOFF_off_b' $p_short
$on_a  = Run-D $false $true  'gOFF_on_a'  $p_short
$on_b  = Run-D $false $true  'gOFF_on_b'  $p_short
$gon_a = Run-D $true  $true  'gON_on_a'   $p_short
$gon_o = Run-D $true  $false 'gON_off_a'  $p_short
Log ""
Log ("short: off repeatability       : " + [FQ]::Bits($off_a, $off_b))
Log ("short: ON vs OFF (graphs off)  : " + [FQ]::Bits($off_a, $on_a))
Log ("short: ON repeatability        : " + [FQ]::Bits($on_a, $on_b))
Log ("short: graphs ON vs graphs OFF : " + [FQ]::Bits($on_a, $gon_a))
Log ("short: graphs ON off vs gOFF off: " + [FQ]::Bits($off_a, $gon_o))

if (Test-Path $p32k) {
    Log ""
    Log "32K prompt:"
    $p32 = @('-c', '40960', '-f', $p32k, '-n', '4')
    $k_o = Run-D $false $false 'gOFF_off_32k' $p32
    $k_n = Run-D $false $true  'gOFF_on_32k'  $p32
    $k_g = Run-D $true  $true  'gON_on_32k'   $p32
    $k_go= Run-D $true  $false 'gON_off_32k'  $p32
    Log ("32k: ON vs OFF (graphs off)    : " + [FQ]::Bits($k_o, $k_n))
    Log ("32k: graphs ON vs graphs OFF   : " + [FQ]::Bits($k_n, $k_g))
    Log ("32k: graphs ON off vs gOFF off : " + [FQ]::Bits($k_o, $k_go))
}
Log ("log: $out")
