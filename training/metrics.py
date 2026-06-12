#!/usr/bin/env python
"""Transliteration metrics: exact-match, CER, Acc@<=1, edit-distance histogram.

Conventions follow the NEWS transliteration shared tasks and the Dakshina paper:
exact-match (Top-1) accuracy is the headline metric, and character error rate
(CER, the normalized edit distance to the nearest reference) is the soft
companion that credits near-misses. ``Acc@<=1`` (within one edit) captures the
"basically right" tail.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein (edit) distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_word(pred: str, refs: Iterable[str]) -> tuple[int, int]:
    """Return (min edit distance to any reference, length of that reference)."""
    best_dist, best_len = None, 0
    for ref in refs:
        d = levenshtein(pred, ref)
        if best_dist is None or d < best_dist:
            best_dist, best_len = d, len(ref)
    return (best_dist if best_dist is not None else 0), best_len


def summarize(dists: list[int], ref_lens: list[int]) -> dict[str, float]:
    """Compute exact-match accuracy, Acc@<=1, and corpus CER."""
    n = max(len(dists), 1)
    total_ref = max(sum(ref_lens), 1)
    return {
        "n": len(dists),
        "acc": 100.0 * sum(d == 0 for d in dists) / n,
        "acc_le1": 100.0 * sum(d <= 1 for d in dists) / n,
        "cer": 100.0 * sum(dists) / total_ref,
    }


def format_summary(stats: dict[str, float], label: str) -> str:
    return (
        f"[{label}]  ACC {stats['acc']:.2f}%   "
        f"Acc@<=1 {stats['acc_le1']:.2f}%   "
        f"CER {stats['cer']:.2f}%   (n={int(stats['n'])})"
    )


def plot_distance_hist(dists: list[int], path: str, title: str, cap: int = 6) -> None:
    """Save a bar chart of the edit-distance distribution (% of words)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(dists), 1)
    counts = Counter(min(d, cap) for d in dists)
    xs = list(range(cap + 1))
    ys = [100.0 * counts.get(x, 0) / n for x in xs]
    labels = [str(x) for x in range(cap)] + [f"{cap}+"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, ys, color="#4C72B0")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("character edit distance to nearest reference (0 = exact match)")
    ax.set_ylabel("% of words")
    ax.set_title(title)
    for x, y in zip(xs, ys, strict=False):
        ax.text(x, y + 0.4, f"{y:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
