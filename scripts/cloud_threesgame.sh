#!/usr/bin/env bash
#
# Parallel play.threesgame.com grind on the 240-core box — the hunt for the 12288
# (two 6144 tiles merging, which ends the game).
#
# WHY THE CLOUD: the 12288 is a ~1.1% event even for deck-aware depth-6 (docs/EXPERIMENTS.md
# H1). On the dev Mac a depth-6 game takes 1-2h and competes with the user's own desktop, so
# ~90 games (the expected number to land one) is a week-plus. Here we run many browser
# sessions at once and get the same 90 games in a day or two.
#
# WHAT WAS VERIFIED BEFORE WRITING THIS (do not re-litigate):
#   * The box has REAL internet: example.com=200, play.threesgame.com resolves to genuine
#     Google IPv6 (it is CNAME'd to c.storage.googleapis.com) and serves a genuine Google
#     cert. There is no corporate MITM here.
#   * `curl https://play.threesgame.com/` fails strict verification with "no alternative
#     certificate subject name matches target host name". That is NOT interception: the site
#     CNAMEs to Google Cloud Storage, which only serves a *.storage.googleapis.com wildcard,
#     so the cert legitimately does not cover play.threesgame.com. It is broken for everyone.
#     threesgame_driver.py already launches Chrome with --ignore-certificate-errors, so this
#     is a non-issue. (`curl -k` returns the real page; plain http:// also returns 200.)
#   * CentOS Stream 9 / glibc 2.34 is fine for Playwright's Chromium (needs >= 2.31).
#
# THE ONE REAL RISK: this box has NO GPU, so headless Chrome renders the game's WebGL through
# SwiftShader (software). That is CPU-hungry and may be slow. Hence --smoke: prove one session
# reaches a real game over here BEFORE burning the box on 24 of them.
#
# Usage:
#   bash scripts/cloud_threesgame.sh --setup            # one-time deps (needs root/sudo)
#   bash scripts/cloud_threesgame.sh --smoke            # 1 session, 1 game — prove it works
#   nohup bash scripts/cloud_threesgame.sh 24 5 > grind.log 2>&1 &   # 24 sessions x 5 games
#   bash scripts/cloud_threesgame.sh --collect          # best game across all sessions
#
# Args (positional): $1=sessions (24)  $2=games-per-session (5)  $3=depth-cap (6)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-9070}"
OUT_ROOT="results/replays/threesgame_cloud"
WORK_ROOT="${WORK_ROOT:-/tmp/tg_cloud}"

# ---------------------------------------------------------------- one-time setup
if [ "${1:-}" = "--setup" ]; then
  # threesgame_driver.py launches with channel="chrome", i.e. REAL Google Chrome — not
  # Playwright's bundled Chromium. So install Chrome itself; its rpm drags in the whole
  # X/nss/cups/gbm dependency set, which is also why we don't hand-list dnf packages
  # (`playwright install-deps` only knows apt distros, not CentOS).
  echo "== Google Chrome (channel=chrome) =="
  sudo dnf install -y https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm || {
    echo "Chrome rpm install failed — check network/sudo, then re-run"; exit 1; }
  google-chrome --version || { echo "chrome not on PATH after install"; exit 1; }
  echo "== Playwright python lib (the API only; Chrome above is the browser) =="
  python3 -m pip install --user playwright
  echo "== sanity: headless Chrome can render on this GPU-less box =="
  python3 - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--ignore-certificate-errors"])
    pg = b.new_page()
    pg.goto("https://play.threesgame.com/", wait_until="domcontentloaded", timeout=60000)
    print("  page title:", pg.title())
    b.close()
print("  OK — Chrome loads the site headless here")
PY
  echo "== done. Next: bash scripts/cloud_threesgame.sh --smoke =="
  exit 0
fi

SESSIONS="${1:-24}"
GAMES="${2:-5}"
DEPTH="${3:-6}"
SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; SESSIONS=1; GAMES=1; DEPTH="${2:-6}"; fi

# ---------------------------------------------------------------- collect mode
if [ "${1:-}" = "--collect" ]; then
  python3 - "$OUT_ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
best, best_p = None, None
n = 0
for p in glob.glob(os.path.join(root, "*", "best.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    n += 1
    if best is None or d.get("final_score", 0) > best.get("final_score", 0):
        best, best_p = d, p
if not best:
    print(f"no best.json under {root}/ yet"); sys.exit(0)
print(f"sessions with a saved game: {n}")
print(f"BEST: {best['final_score']:,} / max {best['max_tile']} / {best['moves']} moves")
print(f"  -> {best_p}  (settlement: {best_p.replace('best.json','best.png')})")
tiles = {}
for p in glob.glob(os.path.join(root, "*", "best.json")):
    try: d = json.load(open(p))
    except Exception: continue
    tiles[d["max_tile"]] = tiles.get(d["max_tile"], 0) + 1
print("max-tile histogram (best per session):",
      " ".join(f"{k}:{v}" for k, v in sorted(tiles.items())))
PY
  exit 0
fi

# ---------------------------------------------------------------- preflight
command -v go >/dev/null || { echo "go not found"; exit 1; }
python3 -c "import playwright" 2>/dev/null || {
  echo "playwright missing — run: bash scripts/cloud_threesgame.sh --setup"; exit 1; }

echo "== building moveserver =="
go build -o bin/moveserver ./cmd/moveserver

# One shared server. -parallelroot=false: with many concurrent sessions the parallelism
# comes from the games, not from splitting each search 4 ways (see ai/search_bb.go).
if ! curl -s "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
  echo "== starting moveserver :$PORT (depth $DEPTH, deck-aware, parallelroot=false) =="
  nohup ./bin/moveserver -addr "127.0.0.1:$PORT" -depthcap "$DEPTH" -deckaware \
    -parallelroot=false > /tmp/moveserver_$PORT.log 2>&1 &
  sleep 3
fi
curl -s "http://127.0.0.1:$PORT/" >/dev/null || { echo "moveserver did not come up"; cat /tmp/moveserver_$PORT.log; exit 1; }

mkdir -p "$OUT_ROOT" "$WORK_ROOT"
echo "== $SESSIONS session(s) x $GAMES game(s), depth $DEPTH, server :$PORT =="
[ "$SMOKE" = 1 ] && echo "== SMOKE: proving ONE session reaches a real game over on this box =="

pids=()
for i in $(seq 0 $((SESSIONS - 1))); do
  # Each session needs its OWN Chrome profile + ply log + record dir: the supervisor
  # resumes a wedged game from that profile's localStorage, so sharing one would make
  # sessions resume each other's boards.
  mkdir -p "$OUT_ROOT/$i"
  # move-timeout 120 < stall-timeout 150: a deep search must not be mistaken for a wedge
  # and killed mid-think (that feedback loop burned 189 restarts on the Mac).
  nohup python3 -W ignore deploy/web/threesgame_supervisor.py \
    --server "http://127.0.0.1:$PORT" \
    --record-dir "$OUT_ROOT/$i" \
    --profile "$WORK_ROOT/prof_$i" \
    --work-dir "$WORK_ROOT/work_$i" \
    --games "$GAMES" --moves 4000 --depth-cap "$DEPTH" \
    --move-timeout 120 --stall-timeout 150 --max-restarts 400 \
    > "$WORK_ROOT/session_$i.log" 2>&1 &
  pids+=($!)
  sleep 0.5      # stagger the Chrome launches
done

echo "== launched ${#pids[@]} session(s). Watch: tail -f $WORK_ROOT/session_0.log =="
echo "== progress:  grep -h 'game .*:' $WORK_ROOT/session_*.log | sort | uniq -c =="
wait "${pids[@]}" || true

echo
echo "== all sessions done =="
grep -h "game .*:" "$WORK_ROOT"/session_*.log 2>/dev/null | sed 's/^/  /' || true
echo
bash "$0" --collect
