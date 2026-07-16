"""AlphaZero-style agent for Threes (single-player STOCHASTIC game), batched for GPU.

Standard AlphaZero targets deterministic two-player games. Threes needs two changes:
(1) decision nodes choose a swipe via PUCT over the policy net; (2) chance nodes model
the random tile spawn — sampled one outcome per simulation (a sampled stochastic MCTS).
Leaves are evaluated by a policy+value net; self-play produces (state, visit-dist,
return) targets.

Efficiency: self-play runs MANY games in parallel and evaluates ALL of a simulation's
leaves (one per active game) in a SINGLE batched forward pass — the difference between
using ~1 GPU core and saturating an H100. `--parallel` sets the batch of games in
flight. `--init` warm-starts from a distilled checkpoint (rl/distill.py) so self-play
begins near depth-D expectimax instead of from scratch.

Run:  python alphazero.py --init models/distilled.pt --parallel 256 --iters 400
The DQN/PPO baselines are separate files and are unaffected by this module.
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from threes_env import ThreesEnv, encode, apply_move, score

VALUE_SCALE = 200_000.0  # return-to-go / SCALE, tanh'd — MATCHES rl/distill.py


class PVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(17, 128, 2)
        self.c2 = nn.Conv2d(128, 128, 2)
        self.fc = nn.Linear(128 * 2 * 2, 256)
        self.pi = nn.Linear(256, 4)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.fc(x.flatten(1)))
        return F.log_softmax(self.pi(x), dim=-1), torch.tanh(self.v(x)).squeeze(-1)


class Node:
    __slots__ = ("P", "N", "W", "children", "legal")

    def __init__(self, prior_legal):
        self.legal = prior_legal              # list of (action, prior)
        self.P = {a: p for a, p in prior_legal}
        self.N = {a: 0 for a, _ in prior_legal}
        self.W = {a: 0.0 for a, _ in prior_legal}
        self.children = {}


def _clone(env, rng):
    """Cheap copy of a game env for a simulation descent, with a fresh rng so each
    simulation samples its own chance (spawn) outcomes."""
    e = ThreesEnv()
    e.board = [row[:] for row in env.board]
    e.bag = env.bag[:]
    e.next, e.next_bonus, e.moves = env.next, env.next_bonus, env.moves
    e.rng = rng
    return e


def _puct(node, c_puct):
    total = sum(node.N.values()) + 1
    sq = math.sqrt(total)
    best_a, best_u = node.legal[0][0], -1e18
    for a, _ in node.legal:
        q = node.W[a] / node.N[a] if node.N[a] > 0 else 0.0
        u = q + c_puct * node.P[a] * sq / (1 + node.N[a])
        if u > best_u:
            best_u, best_a = u, a
    return best_a


def _descend(root, env, c_puct, rng, max_depth=40):
    """Iterative PUCT descent through the (sampled-chance) tree, using the env's REAL
    step dynamics (spawn = previewed next, then a new next). Returns
    (path, leaf_env_or_None, reward_sum): leaf_env is None for a terminal/dead leaf."""
    node = root
    sim = _clone(env, rng)
    path = []          # (node, action, reward_at_edge)
    for depth in range(max_depth + 1):
        if not node.legal:
            return path, None
        a = _puct(node, c_puct)
        _, reward, done, _ = sim.step(a)
        path.append((node, a, reward))
        if done:
            return path, None
        if a not in node.children:      # unexpanded edge -> this is the leaf
            return path, sim
        node = node.children[a]
    return path, None                    # depth cap


def _backup(path, leaf_value):
    """Back up return-to-go: value at each edge = rewards from it to the leaf + V(leaf)."""
    v = leaf_value
    for node, a, r in reversed(path):
        v = r + v
        node.N[a] += 1
        node.W[a] += v


def _net_batch(net, device, obs_list):
    """One forward pass over a list of (17,4,4) obs. Returns (probs[B,4], v[B])."""
    x = torch.from_numpy(np.stack(obs_list)).to(device)
    with torch.no_grad():
        logp, v = net(x)
    return np.exp(logp.cpu().numpy()), v.cpu().numpy()


def batched_selfplay(net, device, n_games, sims, c_puct, seed0, max_moves=20000):
    """Play n_games in parallel; every simulation's leaves (one per active game) are
    evaluated in ONE batched forward pass. Returns (data, finals) where data is a list
    of (obs, pi, score_at_state) and finals is the per-game final score."""
    envs = [ThreesEnv() for _ in range(n_games)]
    for i, e in enumerate(envs):
        e.reset(seed=seed0 + i)
    trajs = [[] for _ in range(n_games)]     # (obs, pi, score_at_state)
    finals = [None] * n_games
    active = [i for i in range(n_games) if envs[i].legal_actions()]
    for i in range(n_games):
        if i not in active:
            finals[i] = envs[i].score()
    sim_ctr = 0

    while active:
        # (1) roots for all active games — one batched eval of the current states
        probs, _ = _net_batch(net, device, [encode(envs[i].board, envs[i].next) for i in active])
        roots = {}
        for k, i in enumerate(active):
            legal = envs[i].legal_actions()
            roots[i] = Node([(a, probs[k][a]) for a in legal])

        # (2) run the simulations, batching leaf evals across games
        for _ in range(sims):
            reqs, terms = [], []
            for i in active:
                rng = random.Random((seed0 + i) * 1_000_003 + sim_ctr)
                sim_ctr += 1
                path, leaf = _descend(roots[i], envs[i], c_puct, rng)
                (reqs if leaf is not None else terms).append((i, path, leaf))
            if reqs:
                probs2, vs = _net_batch(net, device, [encode(le.board, le.next) for _, _, le in reqs])
                for (i, path, le), p, v in zip(reqs, probs2, vs):
                    cl = le.legal_actions()
                    node, a, _ = path[-1]
                    node.children[a] = Node([(aa, p[aa]) for aa in cl]) if cl else None
                    _backup(path, float(v))
            for i, path, _ in terms:
                _backup(path, 0.0)

        # (3) each active game plays its improved-policy move
        nxt_active = []
        for i in active:
            visits = np.array([roots[i].N.get(a, 0) for a in range(4)], dtype=np.float64)
            if visits.sum() == 0:
                finals[i] = envs[i].score()
                continue
            pi = visits / visits.sum()
            trajs[i].append((encode(envs[i].board, envs[i].next), pi.astype(np.float32), envs[i].score()))
            envs[i].step(int(np.random.choice(4, p=pi)))
            if envs[i].moves < max_moves and envs[i].legal_actions():
                nxt_active.append(i)
            else:
                finals[i] = envs[i].score()
        active = nxt_active

    data = []
    for i in range(n_games):
        for obs, pi, s_at in trajs[i]:
            z = math.tanh((finals[i] - s_at) / VALUE_SCALE)   # return-to-go target
            data.append((obs, pi, z))
    return data, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=256, help="games in flight per iter (the GPU batch)")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=2, help="grad epochs over each iter's self-play data")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="models/alphazero.pt")
    ap.add_argument("--init", default="", help="warm-start from a distilled checkpoint (rl/distill.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PVNet().to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"warm-started from {args.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"device {device} | parallel {args.parallel} | sims {args.sims} | "
          f"{sum(p.numel() for p in net.parameters()):,} params", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seed = 10_000_000

    for it in range(1, args.iters + 1):
        net.eval()
        data, finals = batched_selfplay(net, device, args.parallel, args.sims, args.c_puct, seed)
        seed += args.parallel

        loss_val = float("nan")
        if data:
            obs = torch.from_numpy(np.stack([d[0] for d in data]))
            pi = torch.from_numpy(np.stack([d[1] for d in data]))
            z = torch.tensor([d[2] for d in data], dtype=torch.float32)
            net.train()
            idx = np.arange(len(data))
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for s in range(0, len(idx), args.batch):
                    j = idx[s:s + args.batch]
                    xb = obs[j].to(device); pib = pi[j].to(device); zb = z[j].to(device)
                    logp, v = net(xb)
                    loss = -(pib * logp).sum(1).mean() + F.mse_loss(v, zb)
                    opt.zero_grad(); loss.backward(); opt.step()
            loss_val = loss.item()

        print(f"[iter {it:4d}] games={len(finals)} samples={len(data)} "
              f"mean_score={sum(finals)/len(finals):.0f} max={max(finals)} loss={loss_val:.4f}", flush=True)
        torch.save({"model": net.state_dict(), "value_scale": VALUE_SCALE}, args.out)


if __name__ == "__main__":
    main()
