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

T6_BYTES=117275545

if [ ! -f "$IN" ]; then
  # NOT `git lfs pull`: models/ is gitignored and untracked on master, so there is no
  # pointer here to resolve. The checkpoint lives on the stable archive branch, and only
  # a real clone/checkout smudges LFS back into the actual model.
  cat >&2 <<EOF
ERROR: $IN not found — fetch T6's checkpoint from the stable archive:

  rm -rf /tmp/ckpt
  git clone --branch archive/ntuple-checkpoints --single-branch --depth 1 \\
    https://github.com/halfrost/threes-ai.git /tmp/ckpt
  cp /tmp/ckpt/models/ntuple_t6_big2_15m.gob $IN
  ls -la $IN     # MUST be ${T6_BYTES} bytes; ~134 means git-lfs is missing (pointer)
  rm -rf /tmp/ckpt

Needs git-lfs (dnf install -y git-lfs && git lfs install). Do NOT use
'git show <branch>:<file> > $IN' — that writes the LFS POINTER, not the model.
EOF
  exit 1
fi

# Guard the pointer trap: a 134-byte "checkpoint" would otherwise blow up deep inside gob
# decoding (or look like a fresh net) only after the box is committed to a long run.
SZ=$(wc -c < "$IN" | tr -d ' ')
if [ "$SZ" -lt 1000000 ]; then
  echo "ERROR: $IN is only ${SZ} bytes — that is an LFS pointer, not a model." >&2
  echo "       git-lfs did not smudge it. Install git-lfs and re-fetch (see above)." >&2
  exit 1
fi
if [ "$SZ" != "$T6_BYTES" ]; then
  echo "WARNING: $IN is ${SZ} bytes, expected ${T6_BYTES} (T6, big2 15M). Resuming anyway." >&2
fi

echo "Resuming big2+const-alpha=${ALPHA} from ${IN} (${SZ} bytes): +${GAMES} games (seeds from ${SEED}) -> ${OUT}"
./bin/train -resume "$IN" -games "$GAMES" -alpha "$ALPHA" -tuples big2 \
  -train-seed "$SEED" -eval-every 500000 -eval-n 1000 -out "$OUT"

echo "Done. Did big2 overtake big's 21k?"
echo "  python3 scripts/learning_curve.py train_big2_resume.log"
