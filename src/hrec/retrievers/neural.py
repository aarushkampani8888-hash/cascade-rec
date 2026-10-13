"""Neural retrievers: LightGCN (graph) and SASRec (Transformer over sequences).

Both early-stop on the *validation* targets in the tune stage and record the best
epoch; in the test stage they're refit for exactly that many epochs with no access to
any targets. `fit(stage, val_targets=None)` makes that explicit: the pipeline only
ever passes val_targets in the tune stage.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Stage
from ..metrics import mask_seen, target_ranks
from .base import Retriever


def _val_ndcg(model: Retriever, stage: Stage, targets: np.ndarray, k: int = 10) -> float:
    users = np.where(targets >= 0)[0]
    s = mask_seen(model.score(users), stage.matrix(), users)
    r = target_ranks(s, targets[users])
    return float(np.mean(np.where(r < k, 1.0 / np.log2(r + 2), 0.0)))


def _seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================================== LightGCN
class LightGCN(Retriever):
    """He et al., SIGIR 2020. Embeddings smoothed over the user-item graph.

    No feature transforms, no nonlinearities: each layer is E <- Â E with
    Â = D^-1/2 A D^-1/2, and the final embedding is the mean over layers. The point is
    that a user's vector absorbs the items they touched, those items absorb the users
    who touched them, and so on — collaborative signal as message passing.
    """

    name = "lightgcn"

    def __init__(self, dim=64, layers=3, lr=1e-2, reg=1e-4, batch=4096, epochs=None,
                 max_epochs=400, eval_every=10, patience=4, seed=0, verbose=True):
        self.dim, self.layers, self.lr, self.reg, self.batch = dim, layers, lr, reg, batch
        self.epochs, self.max_epochs, self.eval_every, self.patience = epochs, max_epochs, eval_every, patience
        self.seed, self.verbose = seed, verbose
        self.best_epoch = epochs

    def _adj(self, X: sp.csr_matrix) -> torch.Tensor:
        U, I = X.shape
        A = sp.bmat([[None, X], [X.T, None]]).tocoo()
        deg = np.asarray(A.sum(axis=1)).ravel()
        d = np.power(np.maximum(deg, 1e-12), -0.5)
        d[deg == 0] = 0.0
        vals = d[A.row] * A.data * d[A.col]
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32),
                                       (U + I, U + I), check_invariants=False).coalesce()

    def _propagate(self):
        E = self.emb.weight
        out = E
        for _ in range(self.layers):
            E = torch.sparse.mm(self.A, E)
            out = out + E
        out = out / (self.layers + 1)
        return out[: self.n_users], out[self.n_users:]

    def fit(self, stage: Stage, val_targets: np.ndarray | None = None):
        _seed_all(self.seed)
        X = stage.matrix()
        self.n_users, self.n_items = X.shape
        self.A = self._adj(X)
        self.emb = nn.Embedding(self.n_users + self.n_items, self.dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        opt = torch.optim.Adam(self.emb.parameters(), lr=self.lr)
        coo = X.tocoo()
        pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
        rng = np.random.default_rng(self.seed)

        n_epochs = self.epochs if self.epochs is not None else self.max_epochs
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(1, n_epochs + 1):
            perm = rng.permutation(len(pu))
            for s in range(0, len(perm), self.batch):
                b = perm[s: s + self.batch]
                u, i = pu[b], pi[b]
                j = rng.integers(0, self.n_items, size=len(b))
                clash = np.asarray(X[u, j]).ravel() > 0           # resample obvious positives once
                j[clash] = rng.integers(0, self.n_items, size=clash.sum())
                Eu, Ei = self._propagate()
                u_t, i_t, j_t = map(torch.from_numpy, (u, i, j))
                eu, ei, ej = Eu[u_t], Ei[i_t], Ei[j_t]
                loss = -F.logsigmoid((eu * ei).sum(1) - (eu * ej).sum(1)).mean()
                ego = self.emb.weight
                l2 = (ego[u_t].pow(2).sum() + ego[self.n_users + i_t].pow(2).sum()
                      + ego[self.n_users + j_t].pow(2).sum()) / (2 * len(b))
                opt.zero_grad()
                (loss + self.reg * l2).backward()
                opt.step()
            if val_targets is not None and ep % self.eval_every == 0:
                self._cache()
                score = _val_ndcg(self, stage, val_targets)
                if self.verbose:
                    print(f"    lightgcn ep {ep:4d}  val NDCG@10 {score:.4f}  ({time.time()-t0:.0f}s)")
                if score > best:
                    best, bad, self.best_epoch = score, 0, ep
                    best_state = self.emb.weight.detach().clone()
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.emb.weight.data.copy_(best_state)
        self.best_val = best
        self._cache()
        return self

    @torch.no_grad()
    def _cache(self):
        Eu, Ei = self._propagate()
        self.user_vecs, self.item_vecs = Eu.numpy().copy(), Ei.numpy().copy()

    def score(self, users):
        return self.user_vecs[users] @ self.item_vecs.T

    def params(self):
        return {"dim": self.dim, "layers": self.layers, "lr": self.lr, "reg": self.reg,
                "epochs": self.best_epoch}


# =========================================================================== SASRec
class _Block(nn.Module):
    """Pre-LN Transformer block using fused scaled-dot-product attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.drop, self.p = nn.Dropout(dropout), dropout

    def forward(self, x, mask):  # mask: (B, 1, L, L) bool, True = may attend
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class _SASRecNet(nn.Module):
    def __init__(self, n_items, dim, max_len, n_layers, n_heads, dropout):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)  # 0 = padding
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(dim, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq):  # seq: (B, L) left-padded item ids (+1 shifted)
        B, L = seq.shape
        x = self.drop(self.item(seq) * (self.item.embedding_dim ** 0.5) + self.pos.weight[None, -L:])
        # causal + key padding; the diagonal stays open so no row is fully masked (no NaNs)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        mask = (causal[None] & (seq != 0)[:, None, :]) | torch.eye(L, dtype=torch.bool)[None]
        mask = mask[:, None]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)

    def logits(self, h):
        return h @ self.item.weight[1:].T  # tied input/output embeddings


class SASRec(Retriever):
    """Kang & McAuley, ICDM 2018 — causal self-attention over a user's item sequence.

    Trained with full softmax cross-entropy over the catalog rather than the original
    one-negative BCE; Klenitskiy & Vasilev (RecSys 2023) showed that single change is
    worth more than most architecture tweaks. With ~1.7k items the full softmax is cheap.
    """

    name = "sasrec"

    def __init__(self, dim=64, max_len=50, layers=2, heads=2, dropout=0.2, lr=2e-3,
                 weight_decay=0.0, batch=64, epochs=None, max_epochs=300, eval_every=5,
                 patience=6, seed=0, verbose=True):
        self.dim, self.max_len, self.layers, self.heads, self.dropout = dim, max_len, layers, heads, dropout
        self.lr, self.wd, self.batch = lr, weight_decay, batch
        self.epochs, self.max_epochs, self.eval_every, self.patience = epochs, max_epochs, eval_every, patience
        self.seed, self.verbose = seed, verbose
        self.best_epoch = epochs

    def _pad(self, seqs, length):
        out = np.zeros((len(seqs), length), dtype=np.int64)
        for r, s in enumerate(seqs):
            s = s[-length:] + 1
            if len(s):
                out[r, -len(s):] = s
        return out

    def fit(self, stage: Stage, val_targets: np.ndarray | None = None):
        _seed_all(self.seed)
        self.stage = stage
        self.n_items = stage.n_items
        self.net = _SASRecNet(self.n_items, self.dim, self.max_len, self.layers, self.heads, self.dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        seqs = [s for s in stage.sequences() if len(s) >= 2]
        full = self._pad(seqs, self.max_len + 1)  # input = [:-1], target = [1:]
        inp, tgt = torch.from_numpy(full[:, :-1]), torch.from_numpy(full[:, 1:] - 1)  # -1 = pad
        rng = np.random.default_rng(self.seed)

        n_epochs = self.epochs if self.epochs is not None else self.max_epochs
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(1, n_epochs + 1):
            self.net.train()
            perm = torch.from_numpy(rng.permutation(len(inp)))
            for s in range(0, len(perm), self.batch):
                b = perm[s: s + self.batch]
                logits = self.net.logits(self.net(inp[b]))
                loss = F.cross_entropy(logits.reshape(-1, self.n_items), tgt[b].reshape(-1),
                                       ignore_index=-1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if val_targets is not None and ep % self.eval_every == 0:
                score = _val_ndcg(self, stage, val_targets)
                if self.verbose:
                    print(f"    sasrec   ep {ep:4d}  val NDCG@10 {score:.4f}  ({time.time()-t0:.0f}s)")
                if score > best:
                    best, bad, self.best_epoch = score, 0, ep
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val = best
        return self

    @torch.no_grad()
    def user_vectors(self, users) -> np.ndarray:
        self.net.eval()
        seqs = self.stage.sequences()
        x = torch.from_numpy(self._pad([seqs[u] for u in users], self.max_len))
        return self.net(x)[:, -1].numpy()

    def item_vectors(self) -> np.ndarray:
        return self.net.item.weight[1:].detach().numpy()

    def score(self, users):
        out = []
        for s in range(0, len(users), 512):
            out.append(self.user_vectors(users[s: s + 512]) @ self.item_vectors().T)
        return np.vstack(out) if out else np.zeros((0, self.n_items))

    def params(self):
        return {"dim": self.dim, "max_len": self.max_len, "layers": self.layers, "heads": self.heads,
                "dropout": self.dropout, "lr": self.lr, "epochs": self.best_epoch}
