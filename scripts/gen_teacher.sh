#!/usr/bin/env bash
#
# Stage 1 of the AlphaZero distillation pipeline (approach "B"): generate supervised
# training data from the deck-aware expectimax teacher on the 240-core box. Each
# position becomes (board, next, teacher's move, Monte-Carlo return-to-go). The net
# distilled from this (rl/distill.py, on the H100) imitates depth-D expectimax in one
# forward pass — the warm start that lets AlphaZero self-play START near SOTA instead
# of from scratch (the whole reason a from-scratch AZ struggles vs an already-ceiling
# expectimax).
#
# Teacher strength vs cost: d4 ~= 177k mean @360ms/move (strong, the default). d5 is
# stronger (252k) but ~4x slower. On 240 cores, 8000 d4 games ~= a few hours and ~10M
# positions. Scale games for more data.
#
# Usage (on the CPU box):
#   nohup bash scripts/gen_teacher.sh 8000 4 > gen_teacher.log 2>&1 &
#   tail -f gen_teacher.log      # -> data/teacher.bin
# Then copy data/teacher.bin to the H100 box and run rl/distill.py (see rl/README.md).
#
# Args (optional, positional): $1=games (8000)  $2=depth-cap (4)
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-8000}"
DEPTH="${2:-4}"
OUT="data/teacher.bin"

echo "Building gen-teacher..."
go build -o bin/gen-teacher ./cmd/gen-teacher
mkdir -p data

echo "Generating ${GAMES} deck-aware depth-${DEPTH} teacher games -> ${OUT}"
./bin/gen-teacher -n "$GAMES" -depthcap "$DEPTH" -workers "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  -seed 10000000 -out "$OUT"

echo "Done -> ${OUT}. Next: copy to the H100 and distill:"
echo "  python rl/distill.py --data ${OUT} --epochs 30 --batch 4096 --out models/distilled.pt"
echo "  python rl/alphazero.py --init models/distilled.pt --iters 400   # warm-started self-play"
