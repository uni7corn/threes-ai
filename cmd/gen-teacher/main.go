// Command gen-teacher generates supervised distillation data from the strong
// deck-aware expectimax "teacher": it self-plays games and, for every position,
// records (board, next tile, the move the teacher chose, and the Monte-Carlo
// return-to-go under teacher play). A neural net trained on this imitates the
// depth-D expectimax in a single forward pass — the warm start for AlphaZero
// (rl/distill.py consumes this file; then rl/alphazero.py improves it by self-play).
//
// Output is a flat binary stream, 22 bytes per sample, little-endian:
//   [16]uint8 board tile indices (row-major, board[r][c] at r*4+c)
//   uint8     next tile index (previewed)
//   uint8     move (0=UP 1=DOWN 2=LEFT 3=RIGHT)
//   int32     return-to-go = final_score - score_at_this_state
// Python: np.fromfile(path, dtype=[('b','u1',16),('n','u1'),('m','u1'),('r','<i4')]).
//
// Usage:
//   go run ./cmd/gen-teacher -n 5000 -depthcap 4 -workers $(nproc) -out data/teacher.bin
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/halfrost/threes-ai/ai"
	"github.com/halfrost/threes-ai/engine"
	"github.com/halfrost/threes-ai/utils"
)

type sample struct {
	board [16]uint8
	next  uint8
	move  uint8
	score int // teacher score at this state (return-to-go filled in after the game)
}

// playAndCollect runs one deck-aware expectimax game and returns its samples with
// the Monte-Carlo return-to-go computed (final score minus the score at each state).
func playAndCollect(seed int64, maxMoves int) ([]sample, int, int) {
	g := engine.NewGame(seed)
	samples := make([]sample, 0, 1200)
	engine.Play(g, func(gg *engine.Game) int {
		var b [16]uint8
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				b[r*4+c] = uint8(gg.Board[r][c])
			}
		}
		cand := gg.DeckCounts() // deck-aware candidate (true remaining bag)
		mv := ai.ExpectSearchBB(engine.PackBoard(gg.Board), cand, gg.NextHint())
		if mv < 0 {
			return mv // terminal
		}
		samples = append(samples, sample{board: b, next: uint8(gg.Next), move: uint8(mv), score: gg.Score()})
		return mv
	}, nil, maxMoves)
	final := g.Score()
	for i := range samples {
		samples[i].score = final - samples[i].score // return-to-go
	}
	return samples, final, g.MaxTile()
}

func main() {
	n := flag.Int("n", 5000, "number of teacher self-play games")
	seed := flag.Int64("seed", 10_000_000, "base seed (game i uses seed+i; disjoint from eval 1..N)")
	workers := flag.Int("workers", runtime.NumCPU(), "concurrent games")
	depthCap := flag.Int("depthcap", 4, "expectimax depth cap (teacher strength vs speed: d4≈177k mean @360ms/move)")
	maxMoves := flag.Int("maxmoves", 20000, "safety cap on moves per game")
	out := flag.String("out", "data/teacher.bin", "output binary file")
	seqSearch := flag.Bool("seqsearch", true, "run each game's search sequentially (best throughput across many game workers)")
	flag.Parse()

	ai.MaxDepthCap = *depthCap
	ai.ParallelRoot = !*seqSearch
	utils.InitGameScoreTable()

	if dir := filepath.Dir(*out); dir != "" {
		os.MkdirAll(dir, 0o755)
	}
	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create out: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	defer w.Flush()

	var (
		mu          sync.Mutex // guards w
		wg          sync.WaitGroup
		games       int64
		totalSample int64
		sumScore    int64
		start       = time.Now()
		jobs        = make(chan int64, *workers*2)
	)
	fmt.Printf("gen-teacher: %d games, depth-cap %d, %d workers -> %s\n", *n, *depthCap, *workers, *out)

	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 22)
			for s := range jobs {
				samples, final, _ := playAndCollect(s, *maxMoves)
				mu.Lock()
				for _, sm := range samples {
					copy(buf[0:16], sm.board[:])
					buf[16] = sm.next
					buf[17] = sm.move
					binary.LittleEndian.PutUint32(buf[18:22], uint32(int32(sm.score)))
					w.Write(buf)
				}
				mu.Unlock()
				atomic.AddInt64(&totalSample, int64(len(samples)))
				atomic.AddInt64(&sumScore, int64(final))
				done := atomic.AddInt64(&games, 1)
				if done%200 == 0 {
					el := time.Since(start).Seconds()
					fmt.Printf("  %d/%d games | %d samples | mean score %.0f | %.0f games/s\n",
						done, *n, atomic.LoadInt64(&totalSample), float64(atomic.LoadInt64(&sumScore))/float64(done), float64(done)/el)
				}
			}
		}()
	}
	for i := 0; i < *n; i++ {
		jobs <- *seed + int64(i)
	}
	close(jobs)
	wg.Wait()
	w.Flush()

	fi, _ := f.Stat()
	fmt.Printf("Done: %d games, %d samples (%.1f MB), mean teacher score %.0f, %.0fs\n",
		*n, totalSample, float64(fi.Size())/1e6, float64(sumScore)/float64(*n), time.Since(start).Seconds())
}
