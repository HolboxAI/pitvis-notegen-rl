"""Temporal step + instrument recognizer over per-second Endo-FM features.

MS-TCN style: stacked dilated residual 1-D convolutions over the whole video, several
refinement stages, trained with class-balanced cross-entropy for the step, BCE for
instrument presence, and the MS-TCN smoothing loss. num_layers=0 degrades to a
per-second linear probe (the benchmark baseline), which is useful as an ablation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Sequence:
    vid: int
    feats: np.ndarray        # [T, D]
    steps: np.ndarray        # [T]
    instruments: np.ndarray  # [T, K]


class _Residual(nn.Module):
    def __init__(self, dilation, ch, dropout):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.proj(F.relu(self.conv(x))))


class _Stage(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, n_out, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.layers = nn.ModuleList([_Residual(2 ** i, hidden, dropout) for i in range(n_layers)])
        self.out = nn.Conv1d(hidden, n_out, 1)

    def forward(self, x):
        h = self.inp(self.drop(x))
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


class StepTCN(nn.Module):
    def __init__(self, in_dim, n_steps, n_instr, hidden, num_layers, num_stages, dropout):
        super().__init__()
        self.n_steps = n_steps
        n_out = n_steps + n_instr
        n_stages = num_stages if num_layers > 0 else 1
        self.stages = nn.ModuleList(
            [_Stage(in_dim, hidden, num_layers, n_out, dropout)]
            + [_Stage(n_out, hidden, num_layers, n_out, dropout) for _ in range(n_stages - 1)]
        )

    def forward(self, x):  # x: [B, D, T] -> list of [B, S+K, T] logits, one per stage
        outs = [self.stages[0](x)]
        for stage in self.stages[1:]:
            h = outs[-1]
            inp = torch.cat([F.softmax(h[:, :self.n_steps], 1), torch.sigmoid(h[:, self.n_steps:])], 1)
            outs.append(stage(inp))
        return outs


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(train: list, n_steps: int, n_instr: int, mcfg: dict, device, tag: str = "") -> dict:
    """Trains on full-video sequences. Returns a bundle dict (weights + normalization)."""
    _seed(mcfg["seed"])
    all_feats = np.concatenate([s.feats for s in train]).astype(np.float32)
    mean, std = all_feats.mean(0), all_feats.std(0) + 1e-6

    counts = np.bincount(np.concatenate([s.steps for s in train]), minlength=n_steps).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_steps * np.maximum(counts, 1)), 0.0)
    step_w = torch.tensor(np.clip(w, 0, 10), dtype=torch.float32, device=device)
    instr = np.concatenate([s.instruments for s in train]).astype(np.float64)
    pos = instr.sum(0)
    pos_w = torch.tensor(np.clip((len(instr) - pos) / np.maximum(pos, 1), 1, 20),
                         dtype=torch.float32, device=device).view(1, -1, 1)

    model = StepTCN(all_feats.shape[1], n_steps, n_instr, mcfg["hidden"], mcfg["num_layers"],
                    mcfg["num_stages"], mcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    mean_t, std_t = (torch.tensor(a, device=device).view(1, -1, 1) for a in (mean, std))

    for epoch in range(mcfg["epochs"]):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        total = 0.0
        for i in order:
            s = train[i]
            x = (torch.tensor(s.feats.T[None], dtype=torch.float32, device=device) - mean_t) / std_t
            y = torch.tensor(s.steps[None], device=device)
            yi = torch.tensor(s.instruments.T[None], dtype=torch.float32, device=device)
            loss = 0.0
            for out in model(x):
                ls, li = out[:, :n_steps], out[:, n_steps:]
                loss = loss + F.cross_entropy(ls, y, weight=step_w)
                if n_instr:
                    loss = loss + mcfg["instrument_loss_weight"] * F.binary_cross_entropy_with_logits(
                        li, yi, pos_weight=pos_w)
                if ls.shape[-1] > 1:
                    lp = F.log_softmax(ls, 1)
                    loss = loss + mcfg["smoothing_loss_weight"] * torch.clamp(
                        (lp[:, :, 1:] - lp.detach()[:, :, :-1]) ** 2, max=16).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{tag}] epoch {epoch + 1}/{mcfg['epochs']} loss={total / len(train):.4f}")

    return {"state": {k: v.cpu() for k, v in model.state_dict().items()}, "mean": mean, "std": std,
            "in_dim": int(all_feats.shape[1]), "n_steps": n_steps, "n_instr": n_instr,
            "mcfg": {k: mcfg[k] for k in ("hidden", "num_layers", "num_stages", "dropout")}}


@torch.no_grad()
def predict(bundle: dict, feats: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """feats [T, D] -> (step_probs [T, S], instr_probs [T, K])."""
    m = bundle["mcfg"]
    model = StepTCN(bundle["in_dim"], bundle["n_steps"], bundle["n_instr"], m["hidden"],
                    m["num_layers"], m["num_stages"], m["dropout"]).to(device)
    model.load_state_dict(bundle["state"])
    model.eval()
    x = (feats.astype(np.float32) - bundle["mean"]) / bundle["std"]
    out = model(torch.tensor(x.T[None], device=device))[-1][0]
    S = bundle["n_steps"]
    return (F.softmax(out[:S], 0).T.cpu().numpy(),
            torch.sigmoid(out[S:]).T.cpu().numpy())


def estimate_transitions(step_seqs: list, n_steps: int, alpha: float = 1e-3):
    """Per-second log transition matrix + log initial distribution from label sequences.
    Unseen classes get a strong self-transition so Viterbi never strands in them."""
    trans = np.full((n_steps, n_steps), alpha)
    init = np.full(n_steps, alpha)
    for seq in step_seqs:
        if len(seq):
            init[seq[0]] += 1
            np.add.at(trans, (seq[:-1], seq[1:]), 1)
    for i in range(n_steps):
        if trans[i].sum() <= alpha * n_steps + 1e-9:
            trans[i, i] += 1.0
    trans /= trans.sum(1, keepdims=True)
    init /= init.sum()
    return np.log(trans), np.log(init)
