// Command train-ms learns a MULTI-STAGE N-tuple value function for Threes: separate
// networks for separate phases of the game, split by the board's max tile.
//
// Why (docs/EXPERIMENTS.md T7/T8/T9): a single net plateaus at ~23k greedy with 0% at
// 3072, and every single-net lever failed — more games (T7 flat), more capacity (T8 <=
// big), leaf-alignment (T9 collapsed). One weight set has to average over irreconcilable
// regimes: the value of a board changes character once big tiles dominate, so it ends up
// mediocre at all phases. This is exactly the 2016 MS-TD idea (Yeh et al.) that reached
// the 6144 in 7.83% where a plain net got 0.45% — separate value functions per stage.
//
// Learning: afterstate TD(0), but the update touches the stage net of the PREVIOUS
// afterstate and the bootstrap V(s') uses the stage net of the NEXT afterstate, so the
// signal flows correctly across stage boundaries. Move selection evaluates each candidate
// afterstate with its OWN stage's net. Every stage is warm-started from the strongest
// single net (T7's 30M via -init) so the late stages don't start data-starved (early
// self-play rarely reaches them) — they begin at our SOTA and specialise from there.
//
// Usage:
//
//	go run ./cmd/train-ms -init models/ntuple_big.gob -tuples big -stages 10,13 \
//	    -games 20000000 -alpha 0.1 -train-seed 100000000 -eval-every 500000 -out models/ntuple_ms.gob
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/halfrost/threes-ai/engine"
	"github.com/halfrost/threes-ai/gameboard"
	"github.com/halfrost/threes-ai/ntuple"
)

// maxIndex returns the largest tile index on a packed board (0=empty, 10=384, 13=3072…).
func maxIndex(board uint64) int {
	mi := 0
	for c := 0; c < 16; c++ {
		if v := int((board >> (uint(c) * 4)) & 0xF); v > mi {
			mi = v
		}
	}
	return mi
}

// stageOf maps a board to a stage by its max tile index: stage i is entered once the max
// index reaches bounds[i-1]. bounds are ascending tile INDICES; e.g. bounds {10,13} gives
// 3 stages: max<=192 (idx<10) | 384–1536 (10..12) | >=3072 (>=13).
func stageOf(board uint64, bounds []int) int {
	mi := maxIndex(board)
	s := 0
	for _, b := range bounds {
		if mi >= b {
			s++
		} else {
			break
		}
	}
	return s
}

// MS is the multi-stage value function: one N-tuple net per stage.
type MS struct {
	nets    []*ntuple.Network
	bounds  []int
	touched []int64 // per-stage update counts (diagnostic: are the late stages getting data?)
}

func (m *MS) Value(board uint64) float64 { return m.nets[stageOf(board, m.bounds)].Value(board) }

func (m *MS) update(after uint64, target float64, alpha float64) {
	s := stageOf(after, m.bounds)
	m.nets[s].Update(after, alpha*(target-m.nets[s].Value(after)))
	m.touched[s]++
}

// bestMoveMS: greedy over reward + V_stage(afterstate), each candidate dispatched to the
// net of the stage its afterstate lands in.
func bestMoveMS(m *MS, g *engine.Game) (mv int, after uint64, reward int, ok bool) {
	cur := g.Score()
	best := math.Inf(-1)
	mv = -1
	for a := 0; a < 4; a++ {
		nb, _, changeNum := gameboard.MakeMove(g.Board, a)
		if changeNum == 0 {
			continue
		}
		ab := engine.PackBoard(nb)
		r := engine.ScoreBB(ab) - cur
		if eval := float64(r) + m.Value(ab); eval > best {
			best, mv, after, reward = eval, a, ab, r
		}
	}
	return mv, after, reward, mv >= 0
}

// trainGameMS plays one greedy self-play game, updating by afterstate TD(0) with the
// bootstrap V(s') taken from s'’s stage net (m.Value dispatches by stage).
func trainGameMS(m *MS, seed int64, alpha float64, maxMoves int) {
	g := engine.NewGame(seed)
	var prev uint64
	havePrev := false
	for g.Moves < maxMoves {
		mv, after, reward, ok := bestMoveMS(m, g)
		if !ok {
			break
		}
		if havePrev {
			m.update(prev, float64(reward)+m.Value(after), alpha)
		}
		prev, havePrev = after, true
		g.Step(mv)
	}
	if havePrev {
		m.update(prev, 0, alpha) // terminal: no future reward
	}
}

func evalGameMS(m *MS, seed int64, maxMoves int) (score, maxTile int) {
	g := engine.NewGame(seed)
	for g.Moves < maxMoves {
		mv, _, _, ok := bestMoveMS(m, g)
		if !ok {
			break
		}
		g.Step(mv)
	}
	return g.Score(), g.MaxTile()
}

func evalAndReport(m *MS, trained, n, maxMoves int, elapsed time.Duration) {
	scores := make([]int, n)
	var total, maxScore, r3072, r6144 int
	for s := 1; s <= n; s++ {
		score, maxTile := evalGameMS(m, int64(s), maxMoves)
		scores[s-1] = score
		total += score
		if score > maxScore {
			maxScore = score
		}
		if maxTile >= 3072 {
			r3072++
		}
		if maxTile >= 6144 {
			r6144++
		}
	}
	sort.Ints(scores)
	fmt.Printf("[%8d games | %5.0fs] eval(N=%d): mean=%8.0f median=%7d max=%8d | 3072=%4.1f%% 6144=%4.1f%% | touched %v\n",
		trained, elapsed.Seconds(), n, float64(total)/float64(n), scores[n/2], maxScore,
		100*float64(r3072)/float64(n), 100*float64(r6144)/float64(n), m.touched)
}

func parseBounds(s string) []int {
	var b []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad -stages %q: %v\n", s, err)
			os.Exit(1)
		}
		b = append(b, v)
	}
	return b
}

func stageOutPath(out string, i int) string {
	if strings.HasSuffix(out, ".gob") {
		return out[:len(out)-4] + fmt.Sprintf(".stage%d.gob", i)
	}
	return out + fmt.Sprintf(".stage%d.gob", i)
}

func main() {
	games := flag.Int("games", 20_000_000, "self-play training games")
	alpha := flag.Float64("alpha", 0.1, "TD learning rate (constant; T7 proved anneal only hurts)")
	maxMoves := flag.Int("maxmoves", 30000, "safety move cap per game")
	evalEvery := flag.Int("eval-every", 500_000, "evaluate every N training games")
	evalN := flag.Int("eval-n", 1000, "eval games (fixed held-out seeds 1..eval-n)")
	trainSeed := flag.Int64("train-seed", 100_000_000, "base seed for training games (disjoint from eval AND from T1-T9)")
	out := flag.String("out", "models/ntuple_ms.gob", "save prefix; each stage -> <prefix>.stageK.gob")
	initPath := flag.String("init", "", "warm-start EVERY stage from this single net (e.g. T7's 30M models/ntuple_big.gob)")
	tuples := flag.String("tuples", "big", "tuple set if not warm-starting: small|big|big2 (T8: big2's extra capacity is wasted, use big)")
	stagesFlag := flag.String("stages", "10,13", "ascending max-tile INDEX boundaries; default 10,13 => 3 stages (<=192 | 384-1536 | >=3072)")
	flag.Parse()

	bounds := parseBounds(*stagesFlag)
	nStages := len(bounds) + 1

	m := &MS{bounds: bounds, touched: make([]int64, nStages)}
	m.nets = make([]*ntuple.Network, nStages)
	for i := 0; i < nStages; i++ {
		if *initPath != "" {
			// Load the init file once PER stage — each Load is an independent copy, so the
			// stages start identical (at T7's SOTA) and then specialise.
			net, err := ntuple.Load(*initPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "init stage %d from %s: %v\n", i, *initPath, err)
				os.Exit(1)
			}
			m.nets[i] = net
		} else {
			m.nets[i] = ntuple.New(ntuple.TuplesByName(*tuples))
		}
	}

	src := "scratch (" + *tuples + ")"
	if *initPath != "" {
		src = "warm-start " + *initPath
	}
	fmt.Printf("Multi-stage: %d stages, index bounds %v, %s, %d tuples/stage, eval seeds 1..%d (train from %d)\n",
		nStages, bounds, src, len(m.nets[0].Tuples), *evalN, *trainSeed)
	fmt.Printf("Stage map: ")
	labels := []string{}
	prev := 0
	for _, b := range bounds {
		labels = append(labels, fmt.Sprintf("idx[%d,%d)", prev, b))
		prev = b
	}
	labels = append(labels, fmt.Sprintf("idx>=%d", prev))
	fmt.Println(strings.Join(labels, " | "))

	start := time.Now()
	evalAndReport(m, 0, *evalN, *maxMoves, 0) // baseline (the warm-start net dispatched by stage)
	for i := 0; i < *games; i++ {
		trainGameMS(m, *trainSeed+int64(i), *alpha, *maxMoves)
		if (i+1)%*evalEvery == 0 || i+1 == *games {
			evalAndReport(m, i+1, *evalN, *maxMoves, time.Since(start))
			for k := 0; k < nStages; k++ { // checkpoint every stage each eval
				if err := m.nets[k].Save(stageOutPath(*out, k)); err != nil {
					fmt.Fprintf(os.Stderr, "save stage %d: %v\n", k, err)
				}
			}
		}
	}
	fmt.Printf("Done in %.0fs. Stages saved to %s.stage{0..%d}.gob\n",
		time.Since(start).Seconds(), strings.TrimSuffix(*out, ".gob"), nStages-1)
}
