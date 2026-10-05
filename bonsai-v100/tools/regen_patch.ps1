# Regenerate outputs/state-path-fusions.patch as a concatenation of per-file
# `git diff --no-index` runs against the pristine merged tree.
$pristine = if ($env:LLAMA_PRISTINE) { $env:LLAMA_PRISTINE } else { "C:\path\to\pristine-tree\ggml\src\ggml-cuda" }
$tree = if ($env:LLAMA_TREE_GGML_CUDA) { $env:LLAMA_TREE_GGML_CUDA } else { "C:\path\to\llama.cpp-bonsai\ggml\src\ggml-cuda" }
$out = if ($env:PATCH_OUT) { $env:PATCH_OUT } else { "C:\path\to\repo\outputs\state-path-fusions.patch" }
$files = @('common.cuh', 'ggml-cuda.cu', 'gated_delta_net.cu', 'gated_delta_net.cuh',
           'concat.cu', 'concat.cuh', 'ssm-conv.cu', 'ssm-conv.cuh',
           'mmvf.cu', 'mmvf.cuh', 'mmvq.cu', 'mmvq.cuh',
           'unary.cu', 'unary.cuh', 'fwht.cu', 'fwht.cuh', 'vecdotq.cuh')
# git diff --no-index with absolute paths produces headers git apply cannot consume, so
# stage both sides under the same relative layout and diff the two directories: the
# resulting patch applies with `git apply -p1` from the tree root.
$stage = Join-Path $env:TEMP "p1patch"
foreach ($side in 'a', 'b') {
    $dst = Join-Path $stage "$side\ggml\src\ggml-cuda"
    if (Test-Path (Join-Path $stage $side)) { Remove-Item (Join-Path $stage $side) -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
}
foreach ($f in $files) {
    Copy-Item (Join-Path $pristine $f) (Join-Path $stage "a\ggml\src\ggml-cuda\$f") -Force
    Copy-Item (Join-Path $tree $f)     (Join-Path $stage "b\ggml\src\ggml-cuda\$f") -Force
}
Push-Location $stage
# NOTE: capture through cmd's redirection -- PowerShell's Out-String / Set-Content rewrite
# line endings to CRLF (and add a BOM), which makes the patch fail to apply ("?" = CR).
& cmd /c "git diff --no-index --no-prefix -- a b > `"$out`" 2>nul"
Pop-Location
$patchText = Get-Content $out -Raw
foreach ($f in $files) {
    $hit = (Select-String -InputObject $patchText -Pattern ([regex]::Escape("$f")) -AllMatches).Matches.Count
    Write-Output ("{0,-22} mentioned {1}x" -f $f, $hit)
}
$sz = (Get-Item $out).Length
Write-Output "patch: $out  ($sz bytes)"
