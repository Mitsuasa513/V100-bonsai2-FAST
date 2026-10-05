# Bit-exactness of the deinterleaving GLU store (GGML_CUDA_GLU_PERMUTE) against the
# plain RESHAPE -> PERMUTE -> CONT path, both graph modes.
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "dump_glu.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class GCmp {
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
Log "## deinterleaving-GLU-store bit-exactness  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")
function Run-D($graphs, $glu, $tag) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    Remove-Item Env:\GGML_CUDA_GLU_PERMUTE -ErrorAction SilentlyContinue
    if (-not $glu) { $env:GGML_CUDA_GLU_PERMUTE = '0' }
    $f = Join-Path $work "dg_$tag.bin"; if (Test-Path $f) { Remove-Item $f -Force }
    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 4096 -p "The capital of France is" `
        -n 32 --temp 0 --seed 7 --no-display-prompt -co off -st 1> (Join-Path $work "dg_$tag.txt") 2> (Join-Path $work "dg_$tag.err") | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST,Env:\GGML_CUDA_GLU_PERMUTE,Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Log ("  ran {0,-20} dump={1}" -f $tag, (Get-Item $f -ErrorAction SilentlyContinue).Length)
    return $f
}
$off_a = Run-D $false $false 'gOFF_glu_off_a'
$off_b = Run-D $false $false 'gOFF_glu_off_b'
$on_a  = Run-D $false $true  'gOFF_glu_on_a'
$on_b  = Run-D $false $true  'gOFF_glu_on_b'
$gon   = Run-D $true  $true  'gON_glu_on'
Log ""
Log ("glu OFF repeatability      : " + [GCmp]::Bits($off_a, $off_b))
Log ("glu ON  vs glu OFF         : " + [GCmp]::Bits($off_a, $on_a))
Log ("glu ON  repeatability      : " + [GCmp]::Bits($on_a, $on_b))
Log ("graphs ON vs graphs OFF    : " + [GCmp]::Bits($on_a, $gon))
Log ("log: $out")
