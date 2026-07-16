#!/usr/bin/env bash
#
# T7 — resume T2 (`big` 4×6 + CONSTANT alpha) and push it to the true ceiling.
# The 2×2 ablation (T2/T4/T5/T6) settled two things: constant alpha decisively beats
# TC+anneal (the anneal starves late learning), and T2 (big, const-alpha) is the best
# recipe AND was *still climbing* when we stopped it at 10M (+2.5k over its last 3M).
# So the cheapest, highest-confidence strength win is simply MORE games on that recipe.
#
# Loads the 10M checkpoint and trains 20M more games (30M total) on FRESH, disjoint
# self-play seeds. Eval is the same fixed held-out set (seeds 1..1000). `big` is the
# fastest tuple set to train and to evaluate as an expectimax leaf, so if any learned
# value is going to beat the hand heuristic as a leaf, this is the one to re-test (T3)
# — but only if it clears ~25k greedy; below that, weaker leaves already lost.
#
# Seeds: T2 used train seeds 10,000,000 .. 19,999,999 (10M games from the default base).
# We continue at 20,000,000 so no self-play game is replayed.
#
# Usage (detached, survives ssh disconnect):
#   nohup bash scripts/train_big_resume.sh > train_big_resume.log 2>&1 &
#   tail -f train_big_resume.log
#   python3 scripts/learning_curve.py train_big_resume.log     # vs T2's original curve
#
# Args (optional, positional): $1=extra games (20M)  $2=alpha (0.1)  $3=train-seed (20M)
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-20000000}"
ALPHA="${2:-0.1}"
SEED="${3:-20000000}"
IN="models/ntuple_big.gob"          # the T2 checkpoint (~64MB, from cloud-result2)
OUT="models/ntuple_big.gob"         # continue in place (pre-resume ver preserved in git)

echo "Building train..."
go build -o bin/train ./cmd/train

if [ ! -f "$IN" ]; then
  echo "ERROR: $IN not found — need T2's checkpoint to resume. Fetch it first." >&2
  exit 1
fi

echo "Resuming big+const-alpha=${ALPHA} from ${IN}: +${GAMES} games (seeds from ${SEED}) -> ${OUT}"
./bin/train -resume "$IN" -games "$GAMES" -alpha "$ALPHA" -tuples big \
  -train-seed "$SEED" -eval-every 500000 -eval-n 1000 -out "$OUT"

echo "Done. Compare the new asymptote to T2's 21k (@10M):"
echo "  python3 scripts/learning_curve.py train_big_resume.log"
echo "  # if it clears ~25k, re-eval as an expectimax leaf vs the hand heuristic:"
echo "  bash scripts/eval_ntuple_search.sh ${OUT} \$(nproc) \"2 3 4\""
