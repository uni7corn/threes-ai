#!/usr/bin/env bash
#
# T8 — resume T6 (`big2` 8×6 + CONSTANT alpha), the capacity bet.
# In the 2×2 ablation big2 finished −21% below big at 15M — but with the STEEPEST
# still-rising curve of any run (+3.0k over its last 5M). That's the signature of
# under-training, not of a worse model: 2× the weights need far more than 15M games to
# fill in. This resume tests whether big2, given the games, overtakes big's ~21k.
#   * If it clears T2's 21k -> big2 was just under-trained; we have a new best value.
#   * If it plateaus below   -> `big` is the right-sized tuple set for this task.
# (Caveat: big2 is ~2x slower to evaluate as an expectimax leaf, so it must win by a
# real margin to be worth it there.)
#
# Loads the 15M checkpoint (117MB, LFS on cloud-results) and trains 25M more games
# (40M total) on FRESH seeds. T6 used train seeds 10,000,000 .. 24,999,999; we continue
# at 25,000,000. Constant alpha, no TC, no anneal (the ablation's winning levers).
#
# Usage (detached):
#   nohup bash scripts/train_big2_resume.sh > train_big2_resume.log 2>&1 &
#   tail -f train_big2_resume.log
#   python3 scripts/learning_curve.py train_big2_resume.log
#
# Args (optional, positional): $1=extra games (25M)  $2=alpha (0.1)  $3=train-seed (25M)
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-25000000}"
ALPHA="${2:-0.1}"
SEED="${3:-25000000}"
IN="models/ntuple_big2.gob"         # the T6 checkpoint (~117MB, LFS on cloud-results)
OUT="models/ntuple_big2.gob"        # continue in place (pre-resume ver preserved in git)

echo "Building train..."
go build -o bin/train ./cmd/train

if [ ! -f "$IN" ]; then
  echo "ERROR: $IN not found — need T6's checkpoint to resume (git lfs pull)." >&2
  exit 1
fi

echo "Resuming big2+const-alpha=${ALPHA} from ${IN}: +${GAMES} games (seeds from ${SEED}) -> ${OUT}"
./bin/train -resume "$IN" -games "$GAMES" -alpha "$ALPHA" -tuples big2 \
  -train-seed "$SEED" -eval-every 500000 -eval-n 1000 -out "$OUT"

echo "Done. Did big2 overtake big's 21k?"
echo "  python3 scripts/learning_curve.py train_big2_resume.log"
