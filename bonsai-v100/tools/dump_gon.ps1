# Same bit-exact logits dump as dump_matrix.ps1, but with CUDA graphs ENABLED, and
# cross-checked against the graphs-OFF dumps left on disk by that script.

$ErrorActionPreference = 'Continue'

$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out   = Join-Path $work "dump_gon.txt"

$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location $bin

"" | Out-File $out -Encoding utf8
function Log($line) { Write-Host $line; $line | Out-File $out -Append -Encoding utf8 }

Add-Type -TypeDefinition @"
using System;
using System.IO;
public static class FloatCmp3 {
    public static string Compare(string a, string b) {
        if (!File.Exists(a) || !File.Exists(b)) return string.Format("MISSING FILE ({0}, {1})", File.Exists(a), File.Exists(b));
        byte[] ba = File.ReadAllBytes(a); byte[] bb = File.ReadAllBytes(b);
        if (ba.Length != bb.Length) return string.Format("SIZE DIFFERS {0} vs {1}", ba.Length, bb.Length);
        for (int i = 0; i < ba.Length; i++) {
            if (ba[i] != bb[i]) return string.Format("BYTES DIFFER at {0} of {1}", i, ba.Length);
        }
        return string.Format("BYTE-IDENTICAL ({0} bytes)", ba.Length);
    }
}
"@

Log "## dump graphs-ON  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)")

$env:GGML_CUDA_DISABLE_GRAPHS = '0'

function Run-Dump($rows, $tag) {
    if ($rows) { Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ -ErrorAction SilentlyContinue }
    else       { $env:GGML_CUDA_GDN_ROWS_READ = '0' }

    $f = Join-Path $work "go_$tag.bin"
    $t = Join-Path $work "go_$tag.txt"
    $e = Join-Path $work "go_$tag.err"
    foreach ($p in @($f, $t, $e)) { if (Test-Path $p) { Remove-Item $p -Force } }

    $env:GGML_V100_DUMP_LAST = $f
    & (Join-Path $bin 'llama-cli.exe') -m $model -ngl 99 -fa 1 -c 4096 `
        -p "The capital of France is" -n 32 --temp 0 --seed 7 --no-display-prompt -co off -st `
        1> $t 2> $e | Out-Null
    Remove-Item Env:\GGML_V100_DUMP_LAST -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ -ErrorAction SilentlyContinue

    $sz = if (Test-Path $f) { (Get-Item $f).Length } else { 0 }
    Log ("  ran {0,-16} dump={1}" -f $tag, $sz)
    return $f
}

$d1 = Run-Dump 1 'default_a'
$d2 = Run-Dump 1 'default_b'
$r1 = Run-Dump 0 'ref_a'
$r2 = Run-Dump 0 'ref_b'

Log ""
Log "--- graphs ON ---"
Log ("gon default_a vs default_b : " + [FloatCmp3]::Compare($d1, $d2))
Log ("gon ref_a     vs ref_b     : " + [FloatCmp3]::Compare($r1, $r2))
Log ("gon default_a vs ref_a     : " + [FloatCmp3]::Compare($d1, $r1))
Log ""
Log "--- graphs ON vs graphs OFF (dm_* files from dump_matrix.ps1) ---"
Log ("gon default_a vs goff gdn_a     : " + [FloatCmp3]::Compare($d1, (Join-Path $work 'dm_gdn_a.bin')))
Log ("gon default_a vs goff default_a : " + [FloatCmp3]::Compare($d1, (Join-Path $work 'dm_default_a.bin')))
Log ("gon default_a vs goff ref_a     : " + [FloatCmp3]::Compare($d1, (Join-Path $work 'dm_ref_a.bin')))
Log ("gon ref_a     vs goff ref_a     : " + [FloatCmp3]::Compare($r1, (Join-Path $work 'dm_ref_a.bin')))
Log ""
Log ("log: $out")
