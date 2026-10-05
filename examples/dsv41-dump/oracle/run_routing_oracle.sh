#!/bin/bash
# Routing-diagnostic oracle run on astra: reference model.py on CPU (f32, weights dequantized from OUR
# Q8_0 GGUF) with per-layer MoE router capture, on the prompt that pairs with oracle_1199.npz.
# Under the model lock (it reads the 473 GiB GGUF). ~26 GiB RSS for the f32 dense params.
#   usage: run_routing_oracle.sh [out.npz] [prompt file]
set -u
O=~/ml-local/tools/v41-port/oracle
P=${DSV41_ORACLE_PY:-~/ml-local/llama.cpp-v41/.venv/bin/python}
OUT=${1:-$O/oracle_routing_1200.npz}
PF=${2:-$O/prompt1199.txt}
export OMP_NUM_THREADS=64 HF_HUB_OFFLINE=1
echo "######## routing oracle $(basename $PF) -> $OUT  $(date +%T)"
flock -w 14400 ~/ml-local/tools/v41-port/.model.lock \
  $P $O/run_oracle_routing.py --prompt-file $PF --out $OUT \
     --dtype f32 --threads 64 --max-seq-len 2048 --save-layers 0,1,2,14,20,39
echo "rc=$?  $(date +%T)"
ls -la $OUT 2>/dev/null
