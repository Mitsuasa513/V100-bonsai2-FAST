# Isolate which state-path fusion breaks bit-exactness:
#   ref      : GDN_ROWS_READ=0              (both state fusions off, the A/B reference)
#   gdn      : rows read on, conv fusion off
#   gdn+conv : rows read on, conv fusion on
#   default  : everything on
# Each config is run twice so run-to-run repeatability is measured too.

$ErrorActionPreference = 'Continue'

$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "dump_matrix.txt"

$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin

"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }

Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class FloatCmp2 {
    public static string Compare(string a, string b) {
        if (!File.Exists(a) || !File.Exists(b)) return string.Format("MISSING FILE ({0}, {1})", File.Exists(a), File.Exists(b));
        byte[] ba = File.ReadAllBytes(a); byte[] bb = File.ReadAllBytes(b);
        if (ba.Length != bb.Length) return string.Format("SIZE DIFFERS {0} vs {1}", ba.Length, bb.Length);
        int n = ba.Length / 4;
        float[] fa = new float[n]; float[] fb = new float[n];
        Buffer.BlockCopy(ba, 0, fa, 0, ba.Length); Buffer.BlockCopy(bb, 0, fb, 0, bb.Length);
        long nBits = 0; double maxAbs = 0.0; long first = -1; float firstA = 0, firstB = 0;
        long amDiff = 0; int nvocab = 248320;
        for (int i = 0; i < n; i++) {
            int ia = BitConverter.ToInt32(ba, 4 * i), ib = BitConverter.ToInt32(bb, 4 * i);
            if (ia != ib) {
                nBits++;
                double d = Math.Abs((double)fa[i] - (double)fb[i]);
                if (d > maxAbs) maxAbs = d;
                if (first < 0) { first = i; firstA = fa[i]; firstB = fb[i]; }
            }
        }
        int rows = n / nvocab;
        for (int r = 0; r < rows; r++) {
            int am_a = 0, am_b = 0; float va = float.NegativeInfinity, vb = float.NegativeInfinity;
            for (int i = 0; i < nvocab; i++) {
                float x = fa[r * nvocab + i], y = fb[r * nvocab + i];
                if (x > va) { va = x; am_a = i; }
                if (y > vb) { vb = y; am_b = i; }
            }
            if (am_a != am_b) amDiff++;
        }
        return string.Format("floats={0} bitDiff={1} maxAbsDiff={2:E3} firstAt={3} ({4} vs {5}) argmaxDiffer={6}/{7}",
                             n, nBits, maxAbs, first, firstA, firstB, amDiff, rows);
    }
}
"@

Log "## dump matrix  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")

# graphs OFF throughout: we are testing the fusion maths, not the capture path
$env:GGML_CUDA_DISABLE_GRAPHS = '1'

function Run-Dump($rows, $conv, $add, $tag) {
    if ($rows) { Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ -ErrorAction SilentlyContinue }
    else       { $env:GGML_CUDA_GDN_ROWS_READ = '0' }
    if ($conv) { Remove-Item Env:\GGML_CUDA_CONV_STATE_FUSION -ErrorAction SilentlyContinue }
    else       { $env:GGML_CUDA_CONV_STATE_FUSION = '0' }
    if ($add)  { Remove-Item Env:\GGML_CUDA_ADD_RMS_NORM -ErrorAction SilentlyContinue }
    else       { $env:GGML_CUDA_ADD_RMS_NORM = '0' }

    $f = Join-Path $work "dm_$tag.bin"
    $t = Join-Path $work "dm_$tag.txt"
    $e = Join-Path $work "dm_$tag.err"
    foreach ($p in @($f, $t, $e)) { if (Test-Path $p) { Remove-Item $p -Force } }

    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 4096 `
        -p "The capital of France is" -n 32 --temp 0 --seed 7 --no-display-prompt -co off -st `
        1> $t 2> $e | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ,Env:\GGML_CUDA_CONV_STATE_FUSION,Env:\GGML_CUDA_ADD_RMS_NORM -ErrorAction SilentlyContinue

    $sz = if (Test-Path $f) { (Get-Item $f).Length } else { 0 }
    Log ("  ran {0,-14} dump={1}" -f $tag, $sz)
    return $f
}

$ref_a = Run-Dump 0 0 1 'ref_a'
$ref_b = Run-Dump 0 0 1 'ref_b'
$gdn_a = Run-Dump 1 0 1 'gdn_a'
$gdn_b = Run-Dump 1 0 1 'gdn_b'
$cnv_a = Run-Dump 1 1 1 'conv_a'
$cnv_b = Run-Dump 1 1 1 'conv_b'
$def_a = Run-Dump 1 1 1 'default_a'
$def_b = Run-Dump 1 1 1 'default_b'

Log ""
Log "--- repeatability of the same config ---"
Log ("ref_a     vs ref_b     : " + [FloatCmp2]::Compare($ref_a, $ref_b))
Log ("gdn_a     vs gdn_b     : " + [FloatCmp2]::Compare($gdn_a, $gdn_b))
Log ("conv_a    vs conv_b    : " + [FloatCmp2]::Compare($cnv_a, $cnv_b))
Log ("default_a vs default_b : " + [FloatCmp2]::Compare($def_a, $def_b))
Log ""
Log "--- against the reference (both state fusions off) ---"
Log ("gdn_a     vs ref_a     : " + [FloatCmp2]::Compare($ref_a, $gdn_a))
Log ("conv_a    vs ref_a     : " + [FloatCmp2]::Compare($ref_a, $cnv_a))
Log ("default_a vs ref_a     : " + [FloatCmp2]::Compare($ref_a, $def_a))
Log ""
Log ("log: $out")
