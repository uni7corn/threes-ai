"""Distill the deck-aware expectimax teacher into a policy+value net.

Supervised warm-start for AlphaZero (and a fast strong leaf in its own right):
reads the binary produced by `cmd/gen-teacher` — (board, next, teacher move,
Monte-Carlo return-to-go) — and trains the SAME PVNet as alphazero.py, so the
checkpoint loads directly as an AlphaZero starting point.

  policy head: cross-entropy toward the teacher's move (the main signal — a net
               that mimics depth-D expectimax in one forward pass).
  value  head: MSE toward tanh(return_to_go / VALUE_SCALE) — matches PVNet's tanh
               value; invert as atanh(v)*VALUE_SCALE to read a score-scale leaf value.

Usage (H100):
  python rl/distill.py --data data/teacher.bin --epochs 30 --batch 4096 --out models/distilled.pt
Then warm-start AlphaZero:
  python rl/alphazero.py --init models/distilled.pt --iters 400     # (--init to be added)
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

# Reuse the EXACT net from alphazero.py so the checkpoint is a drop-in warm start.
from alphazero import PVNet

VALUE_SCALE = 200_000.0  # return-to-go ~[0,2M]; /200k keeps tanh in a resolving range
TEACHER_DT = np.dtype([('b', 'u1', 16), ('n', 'u1'), ('m', 'u1'), ('r', '<i4')])


def encode_batch(board, nxt, device):
    """(B,16) indices + (B,) next -> (B,17,4,4): 16 one-hot tile-index channels + next."""
    b = torch.as_tensor(board, dtype=torch.long, device=device)           # (B,16)
    oh = F.one_hot(b, num_classes=16).float()                             # (B,16,16) cell×chan
    oh = oh.view(-1, 4, 4, 16).permute(0, 3, 1, 2).contiguous()           # (B,16,4,4)
    n = torch.as_tensor(nxt, dtype=torch.float32, device=device).view(-1, 1, 1, 1) / 15.0
    nch = n.expand(-1, 1, 4, 4)                                           # (B,1,4,4)
    return torch.cat([oh, nch], dim=1)                                    # (B,17,4,4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/teacher.bin", help="cmd/gen-teacher binary")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--out", default="models/distilled.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    d = np.fromfile(a.data, dtype=TEACHER_DT)
    if len(d) == 0:
        raise SystemExit(f"no samples in {a.data}")
    print(f"loaded {len(d):,} teacher samples from {a.data}")

    board, nxt, move = d['b'], d['n'], d['m']
    vtar = np.tanh(d['r'].astype(np.float32) / VALUE_SCALE)               # value target

    n_val = int(len(d) * a.val_frac)
    perm = np.random.default_rng(0).permutation(len(d))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    net = PVNet().to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"device {a.device} | {sum(p.numel() for p in net.parameters()):,} params | "
          f"train {len(tr_idx):,} val {len(val_idx):,}")

    def run_batches(idx, train):
        net.train(train)
        tot_p = tot_v = tot_acc = 0.0
        nb = 0
        for s in range(0, len(idx), a.batch):
            j = idx[s:s + a.batch]
            x = encode_batch(board[j], nxt[j], a.device)
            mt = torch.as_tensor(move[j], dtype=torch.long, device=a.device)
            vt = torch.as_tensor(vtar[j], dtype=torch.float32, device=a.device)
            logp, v = net(x)
            lp = F.nll_loss(logp, mt)
            lv = F.mse_loss(v, vt)
            loss = lp + a.value_weight * lv
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_p += lp.item(); tot_v += lv.item()
            tot_acc += (logp.argmax(1) == mt).float().mean().item()
            nb += 1
        return tot_p / nb, tot_v / nb, tot_acc / nb

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(1, a.epochs + 1):
        np.random.shuffle(tr_idx)
        p, v, acc = run_batches(tr_idx, True)
        with torch.no_grad():
            vp, vv, vacc = run_batches(val_idx, False)
        print(f"epoch {ep:2d} | train policy_nll {p:.3f} val_mse {v:.4f} move_acc {acc:.3f} "
              f"|| val policy_nll {vp:.3f} val_mse {vv:.4f} move_acc {vacc:.3f}")
        if vacc > best:
            best = vacc
            torch.save({"model": net.state_dict(), "value_scale": VALUE_SCALE,
                        "val_move_acc": vacc}, a.out)
    print(f"done. best val move-acc {best:.3f} -> {a.out} (warm-start alphazero with --init {a.out})")


if __name__ == "__main__":
    main()
