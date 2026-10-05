# Multi-sequence acceptance (v2): two concurrent greedy completions via llama-server -np 2,
# capturing the response bodies and the CUDA node dump, so that
#   (a) the dump proves a batch with n_seqs = 2 was executed (GET_ROWS ne[1] == 2), and
#   (b) the generated text is byte-compared between rows and gathered.

$ErrorActionPreference = 'Continue'

$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "C:\path\to\llama.cpp-bonsai\build-v100\bin" }
$model = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL } else { "C:\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" }
$work = if ($env:LLAMA_WORK) { $env:LLAMA_WORK } else { "C:\path\to\work" }
$out    = Join-Path $work "multiseq2_results.txt"
$server = Join-Path $bin "llama-server.exe"
$port   = 8129

$env:CUDA_VISIBLE_DEVICES = '0'
$env:GGML_V100_DUMP_NODES = '1'
Set-Location $bin

function Log($line) { Write-Output $line; $line | Out-File $out -Append -Encoding utf8 }

$body1 = '{"prompt":"The capital of France is","n_predict":32,"temperature":0,"seed":7,"cache_prompt":false}'
$body2 = '{"prompt":"The capital of Japan is","n_predict":32,"temperature":0,"seed":7,"cache_prompt":false}'

function Run-Case($graphs, $rows, $tag) {
    if ($graphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '0' } else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
    if ($rows)   { Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ -ErrorAction SilentlyContinue }
    else         { $env:GGML_CUDA_GDN_ROWS_READ = '0' }

    $srvErr = Join-Path $work "ms2_${tag}.err"
    $srvOut = Join-Path $work "ms2_${tag}.out"
    $resp1  = Join-Path $work "ms2_${tag}_1.json"
    $resp2  = Join-Path $work "ms2_${tag}_2.json"

    $proc = Start-Process -FilePath $server -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $srvOut -RedirectStandardError $srvErr `
        -ArgumentList @('-m', $model, '-ngl', '99', '-fa', '1', '-c', '4096', '-np', '2',
                        '--host', '127.0.0.1', '--port', "$port", '-b', '2048', '-ub', '2048')

    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Milliseconds 1000
        try { if ((Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ready = $true; break } } catch { }
        if ($proc.HasExited) { break }
    }
    if (-not $ready) {
        Log "  [$tag] server not ready"
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
        return $null
    }

    $j1 = Start-Job -ScriptBlock {
        param($b, $p, $o)
        Invoke-WebRequest -Uri "http://127.0.0.1:$p/completion" -Method POST -Body $b `
            -ContentType 'application/json' -UseBasicParsing -TimeoutSec 120 |
            Select-Object -ExpandProperty Content | Set-Content -LiteralPath $o -Encoding utf8
    } -ArgumentList $body1, $port, $resp1
    $j2 = Start-Job -ScriptBlock {
        param($b, $p, $o)
        Invoke-WebRequest -Uri "http://127.0.0.1:$p/completion" -Method POST -Body $b `
            -ContentType 'application/json' -UseBasicParsing -TimeoutSec 120 |
            Select-Object -ExpandProperty Content | Set-Content -LiteralPath $o -Encoding utf8
    } -ArgumentList $body2, $port, $resp2

    $null = Wait-Job $j1, $j2 -Timeout 300
    Receive-Job $j1, $j2 | Out-Null
    Remove-Job $j1, $j2 -Force

    Start-Sleep -Milliseconds 800
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 800

    $t1 = ''; $t2 = ''
    try { $t1 = (Get-Content -LiteralPath $resp1 -Raw | ConvertFrom-Json).content } catch { $t1 = '' }
    try { $t2 = (Get-Content -LiteralPath $resp2 -Raw | ConvertFrom-Json).content } catch { $t2 = '' }

    # dump evidence: state gather with two rows == a 2-sequence batch went through the graph
    $b2 = (Select-String -Path $srvErr -Pattern 'GET_ROWS\s+\S+\s+ne=\(786432,2,' -ErrorAction SilentlyContinue).Count
    $rowdbg = (Select-String -Path $srvErr -Pattern '\[gdn-rows\]' -ErrorAction SilentlyContinue).Count

    Log ("  [{0}] seq1_len={1} seq2_len={2} | batches_with_n_seqs=2: {3} | gdn-rows merges: {4}" -f
         $tag, $t1.Length, $t2.Length, $b2, $rowdbg)
    return @{ t1 = $t1; t2 = $t2; b2 = $b2 }
}

"" | Out-File $out -Encoding utf8
"## multi-sequence acceptance (v2)  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $out -Append -Encoding utf8
"dll: $((Get-Item (Join-Path $bin 'ggml-cuda.dll')).LastWriteTime)" | Out-File $out -Append -Encoding utf8

foreach ($g in @($true, $false)) {
    $gt = if ($g) { 'gON' } else { 'gOFF' }
    Log ""
    Log "=== $gt ==="
    $r1 = Run-Case $g $true  "${gt}_rows"
    $r2 = Run-Case $g $false "${gt}_gather"
    if ($r1 -and $r2) {
        Log ("  [$gt] seq1 text: " + $(if ($r1.t1 -ceq $r2.t1 -and $r1.t1.Length -gt 0) { 'IDENTICAL' } else { '*** DIFFERS ***' }))
        Log ("  [$gt] seq2 text: " + $(if ($r1.t2 -ceq $r2.t2 -and $r1.t2.Length -gt 0) { 'IDENTICAL' } else { '*** DIFFERS ***' }))
        Log ("  [$gt] rows   : " + ($r1.t1 + ' | ' + $r1.t2))
        Log ("  [$gt] gather : " + ($r2.t1 + ' | ' + $r2.t2))
    }
}

Remove-Item Env:\GGML_CUDA_GDN_ROWS_READ -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_V100_DUMP_NODES -ErrorAction SilentlyContinue
Log ""
Log "results: $out"
