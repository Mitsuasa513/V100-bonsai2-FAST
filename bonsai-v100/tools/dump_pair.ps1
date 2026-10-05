# Bit-exactness of the ssm_alpha/ssm_beta matvec pair fusion, against the pure baseline
# (state fusions off + pair off).
$ErrorActionPreference = 'Continue'
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "dump_pair.txt"
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin
"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }
Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class PC {
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
    public static bool Same(string a, string b) { return Bits(a, b).StartsWith("BIT-IDENTICAL"); }
}
"@
Log "## matvec-pair bit-exactness  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")
function Run-D($graphs, $state, $pair, $tag) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ,Env:\GGML_CUDA_MMVF_PAIR -ErrorAction SilentlyContinue
    if (-not $state) { $env:GGML_CUDA_GDN_ROWS_READ = '0' }
    if (-not $pair)  { $env:GGML_CUDA_MMVF_PAIR = '0' }
    $f = Join-Path $work "dp_$tag.bin"; if (Test-Path $f) { Remove-Item $f -Force }
    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 4096 -p "The capital of France is" `
        -n 32 --temp 0 --seed 7 --no-display-prompt -co off -st 1> (Join-Path $work "dp_$tag.txt") 2> (Join-Path $work "dp_$tag.err") | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST,Env:\GGML_CUDA_GDN_ROWS_READ,Env:\GGML_CUDA_MMVF_PAIR,Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Log ("  ran {0,-22} dump={1}" -f $tag, (Get-Item $f -ErrorAction SilentlyContinue).Length)
    return $f
}
$base = Run-D $false $false $false 'gOFF_stateOFF_pairOFF'
$p1   = Run-D $false $true  $false 'gOFF_stateON_pairOFF'
$p2   = Run-D $false $true  $true  'gOFF_stateON_pairON'
$p3   = Run-D $false $true  $true  'gOFF_stateON_pairON_b'
$p4   = Run-D $true  $true  $true  'gON_stateON_pairON'
$p5   = Run-D $false $false $true  'gOFF_stateOFF_pairON'
Log ""
Log ("pair ON  vs pair OFF (state fusions on)  : " + [PC]::Bits($p1, $p2))
Log ("pair ON repeatability                    : " + [PC]::Bits($p2, $p3))
Log ("pair ON (state off) vs pure baseline     : " + [PC]::Bits($base, $p5))
Log ("everything vs pure baseline              : " + [PC]::Bits($base, $p2))
Log ("graphs ON vs graphs OFF (all on)         : " + [PC]::Bits($p2, $p4))
Log ("log: $out")
