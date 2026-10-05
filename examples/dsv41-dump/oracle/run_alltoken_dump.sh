#!/bin/bash
# All-token logits + MoE router dump of the prompt that pairs with oracle_1199.npz (prompt1199.txt: 1199 text
# tokens + BOS = 1200 positions), with the v41-dump-logits build (examples/dsv41-dump with DSV41_DUMP_ALL_LOGITS).
# CPU-only (-dev none), single ubatch (-ub 2048 > 1200 tokens), under the model lock.
#   usage: run_alltoken_dump.sh [outdir] [prompt file]
set -u
O=~/ml-local/tools/v41-port/oracle
G=~/ml-local/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-Q8_0.gguf
B=${DSV41_DUMP_BIN:-~/ml-local/wt/dump-logits/build-cpu/bin/llama-dsv41-dump}
D=${1:-$O/dump_alltok_1200}
PF=${2:-$O/prompt1199.txt}
export CUDA_VISIBLE_DEVICES="" LLAMA_MMAP_NO_PREFETCH=1 DSV41_FORCE_BOS=1 DSV41_DUMP_ALL_LOGITS=1
export DSV41_ENGRAM_CONSTS=$O/engram_consts
export DSV41_DUMP_FILTER='^(l_last-[0-9]+|result_output|ffn_moe_topk-[0-9]+|ffn_moe_probs-[0-9]+|ffn_moe_probs_biased-[0-9]+)$'
rm -rf "$D"; mkdir -p "$D"
echo "######## all-token dump of $(basename $PF) -> $D  $(date +%T)"
DSV41_DUMP_DIR=$D flock -w 14400 ~/ml-local/tools/v41-port/.model.lock \
  $B -m $G -f $PF -dev none -lm mmap --no-host --no-repack -t 64 -c 4096 -b 2048 -ub 2048 > $D/run.log 2>&1
echo "rc=$?  files=$(ls $D | wc -l)  $(date +%T)"
grep -aE "dsv41-dump: [0-9]+ tokens|wrote [0-9]+ tensors|expected|error|Assert|abort" $D/run.log | head
