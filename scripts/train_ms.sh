#!/usr/bin/env bash
#
# T10 — MULTI-STAGE N-tuple TD (approach the 2016 MS-TD SOTA).
#
# The whole single-net line is capped (docs/EXPERIMENTS.md): more games (T7), more
# capacity (T8), and leaf-alignment (T9) all failed — one net tops out at ~23-24k greedy
# and 0.0% at 3072 because it must average over irreconcilable game phases. Yeh et al.'s
# MS-TD reached the 6144 in 7.83% (vs 0.45% plain) by giving each PHASE its own value
# function. This trains exactly that: separate `big` nets split by the board's max tile,
# each warm-started from T7's 30M net so the mid/late stages don't start data-starved.
#
# WATCH THE `touched` COUNTS in the eval line — they show updates per stage. If the top
# stage stays at ~0, self-play isn't reaching it; lower its boundary (e.g. -stages 10,12)
# so it gets data, or expect the gain to come first from the lower stages specialising.
#
# Two variants worth running in parallel on the two free training boxes:
#   machine A:  bash scripts/train_ms.sh                 # 3 stages: <=192 | 384-1536 | >=3072
#   machine B:  STAGES=9,11,13 bash scripts/train_ms.sh  # 4 stages: <=96 | 192-768 | 1536 | >=3072
#
# Usage (detached):
#   nohup bash scripts/train_ms.sh > train_ms.log 2>&1 &
#   tail -f train_ms.log
# Args (optional, positional): $1=games (20M)  $2=alpha (0.1)
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-20000000}"
ALPHA="${2:-0.1}"
STAGES="${STAGES:-10,13}"
SEED="${SEED:-100000000}"          # disjoint from T1-T9 (which used <=90M+)
INIT="${INIT:-models/ntuple_big.gob}"   # T7's 30M net (strongest single net)
OUT="${OUT:-models/ntuple_ms.gob}"
T7_BYTES=72702310

echo "Building train-ms..."
go build -o bin/train-ms ./cmd/train-ms

if [ ! -f "$INIT" ]; then
  cat >&2 <<EOF
ERROR: warm-start net $INIT not found — fetch T7's 30M checkpoint from the stable archive:

  rm -rf /tmp/ckpt
  git clone --branch archive/ntuple-checkpoints --single-branch --depth 1 \\
    https://github.com/halfrost/threes-ai.git /tmp/ckpt
  cp /tmp/ckpt/models/ntuple_t7_big_30m.gob $INIT
  ls -la $INIT     # MUST be ${T7_BYTES} bytes; ~134 means git-lfs is missing (pointer)
  rm -rf /tmp/ckpt

Needs git-lfs (dnf install -y git-lfs && git lfs install).
EOF
  exit 1
fi

SZ=$(wc -c < "$INIT" | tr -d ' ')
if [ "$SZ" -lt 1000000 ]; then
  echo "ERROR: $INIT is only ${SZ} bytes — that's an LFS pointer, not a model. Re-fetch (see above)." >&2
  exit 1
fi
[ "$SZ" = "$T7_BYTES" ] || echo "WARNING: $INIT is ${SZ} bytes, expected ${T7_BYTES} (T7 30M). Proceeding." >&2

echo "T10 multi-stage: warm-start ${INIT} (${SZ} bytes), stages=${STAGES}, big, alpha=${ALPHA}, ${GAMES} games, seeds from ${SEED} -> ${OUT}"
./bin/train-ms -init "$INIT" -tuples big -stages "$STAGES" \
  -games "$GAMES" -alpha "$ALPHA" -train-seed "$SEED" \
  -eval-every 500000 -eval-n 1000 -out "$OUT"

echo "Done. Compare to the single-net ceiling: T7 big 23,507 / 0.0% @3072."
echo "Stage files: ${OUT%.gob}.stage*.gob"
