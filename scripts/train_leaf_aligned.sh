#!/usr/bin/env bash
#
# T9 — leaf-aligned fine-tuning (approach "A"): make the N-tuple value a good SEARCH
# LEAF instead of a good greedy player. T3 found the puzzle: T2 (the strongest greedy
# value) BEATS the hand heuristic as a depth-3 expectimax leaf but LOSES at d4-d5. The
# hypothesis: it was trained by greedy (depth-0) self-play, so it estimates the return
# under *greedy* play — miscalibrated for use as a *search* leaf.
#
# Fix: fine-tune it under self-play where every move is chosen by a depth-D expectimax
# with the net itself at the leaves (`-search-depth D`). Now the value learns the return
# under *search* play — exactly how it is used. Warm-started from T2 so we only re-align
# a good value (few games, small alpha), not learn from scratch.
#
# Cost: depth-D self-play evaluates the net over a d-ply expansion each move, so it is
# ~1-2 orders of magnitude slower per game than greedy. Single-threaded (the search
# fans out 4 ways internally); ~20-30 games/s at D=1. 1.5M games ~= 12-20h on the box.
# NOTE: the per-move eval during training is still GREEDY (comparable to T2's curve) and
# is EXPECTED to dip a little — the value is re-aiming from greedy to search. The real
# test is the leaf eval below.
#
# Usage (detached):
#   nohup bash scripts/train_leaf_aligned.sh > train_leaf.log 2>&1 &
#   tail -f train_leaf.log
# Then the decisive test — is it a better leaf than greedy-trained T2? (compare to T3):
#   bash scripts/eval_ntuple_search.sh models/ntuple_big_leaf.gob $(nproc) "2 3 4 5"
#   # T3 baseline (greedy-trained T2 as leaf, deck-aware, N=1000):
#   #   d3 126,952 | d4 159,011 | d5 190,246   (hand heuristic: d4 177,042, d5 251,707)
#   # A "wins" if the leaf-aligned model closes the d4/d5 gap to the hand heuristic.
#
# Args (optional, positional): $1=games (1.5M)  $2=alpha (0.03)  $3=search-depth (1)
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-1500000}"
ALPHA="${2:-0.03}"
DEPTH="${3:-1}"
IN="models/ntuple_big.gob"          # warm-start from T2 (best greedy value, ~64MB)
OUT="models/ntuple_big_leaf.gob"    # new file — keep T2's greedy model intact

echo "Building train..."
go build -o bin/train ./cmd/train

if [ ! -f "$IN" ]; then
  echo "ERROR: $IN (T2 checkpoint) not found — fetch it first." >&2
  exit 1
fi

echo "Leaf-aligned fine-tune: resume ${IN}, big, alpha=${ALPHA}, search-depth=${DEPTH}, ${GAMES} games -> ${OUT}"
./bin/train -resume "$IN" -games "$GAMES" -alpha "$ALPHA" -tuples big \
  -search-depth "$DEPTH" -train-seed 90000000 -eval-every 100000 -eval-n 1000 -out "$OUT"

echo "Done. Decide it as a LEAF (the whole point) vs the hand heuristic:"
echo "  bash scripts/eval_ntuple_search.sh ${OUT} \$(nproc) \"2 3 4 5\""
