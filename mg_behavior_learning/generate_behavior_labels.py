#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_behavior_labels.py

Phase-1 rule-based behavior labels for MarioDataset slices.
Output: behavior_labels.pt (torch.LongTensor of shape [len(dataset)]), values in [0..9].

Usage examples:
  python generate_behavior_labels.py \
      --output ./Mario-GPT2-700-context-length/behavior_labels.pt

  # If you want to use a custom level txt:
  python generate_behavior_labels.py --level_string ./my_level.txt --output behavior_labels.pt

Notes:
- This script assumes dataset slice corresponds to 14(height) x (context_len/14) columns layout,
  matching how MarioDataset tokenizes the level string.
- Labels are heuristics for Phase 1. Later you can replace them with MCTS/video-derived labels.
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_behavior_learning.dataset import MarioDataset

# ---------------------------
# Behavior label dictionary
# ---------------------------
BEHAVIORS = {
    0: "FLOW_FAST",
    1: "RHYTHM_STUTTER",
    2: "PREC_HI",
    3: "PREC_LO",
    4: "JUMP_ARC_MAX",
    5: "JUMP_STREAK",
    6: "GAP_BLIND",
    7: "FAIL_30",
    8: "FAIL_50",
    9: "NORMAL",
}


# ---------------------------
# Helpers: slice -> 2D grid
# ---------------------------
def tokens_to_grid(tokens: torch.Tensor, tokenizer, height: int = 14):
    """
    Convert a 1D token slice into a 2D char grid [H, W].
    Assumes decoding yields a string with length divisible by height (or will be truncated).
    """
    s = tokenizer.decode(tokens.detach().cpu())
    if len(s) < height:
        # degenerate
        return np.full((height, 1), "-", dtype="<U1")

    w = len(s) // height
    s = s[: w * height]
    cols = [list(s[i * height : (i + 1) * height]) for i in range(w)]
    grid = np.array(cols, dtype="<U1").T  # [H, W]
    return grid


# ---------------------------
# Simple geometry features
# ---------------------------
def ground_row(height: int = 14):
    # In SMB representation, bottom row often contains ground 'X'.
    # Many datasets use row height-1 as solid ground.
    return height - 1


def is_solid(ch: str):
    # Treat these as solid-ish for platforming.
    # Adjust if your tiles differ.
    return ch in {"X", "S", "?", "Q", "B", "b", "<", ">", "[", "]"}


def col_solid_height(grid: np.ndarray, col: int):
    """Return the highest (smallest row index) solid tile in this column, or None if none."""
    h = grid.shape[0]
    for r in range(h):
        if is_solid(grid[r, col]):
            return r
    return None


def bottom_is_gap(grid: np.ndarray, col: int):
    """True if bottom cell is not solid."""
    r = grid.shape[0] - 1
    return not is_solid(grid[r, col])


def count_gaps_on_bottom(grid: np.ndarray):
    return sum(bottom_is_gap(grid, c) for c in range(grid.shape[1]))


def longest_gap_run(grid: np.ndarray):
    """Longest consecutive run of bottom gaps."""
    best = cur = 0
    for c in range(grid.shape[1]):
        if bottom_is_gap(grid, c):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def min_platform_width(grid: np.ndarray):
    """
    Approximate minimal contiguous platform width on bottom row.
    If no solid at bottom, return 0.
    """
    r = grid.shape[0] - 1
    solid = [1 if is_solid(grid[r, c]) else 0 for c in range(grid.shape[1])]
    widths = []
    cur = 0
    for v in solid:
        if v == 1:
            cur += 1
        else:
            if cur > 0:
                widths.append(cur)
            cur = 0
    if cur > 0:
        widths.append(cur)
    return min(widths) if widths else 0


def count_enemies(grid: np.ndarray):
    return int(np.sum(grid == "E"))


def count_vertical_structure_changes(grid: np.ndarray):
    """
    Rough proxy for 'stuttery' terrain: count changes in solid height from col to col.
    """
    hs = []
    for c in range(grid.shape[1]):
        r = col_solid_height(grid, c)
        hs.append(r if r is not None else grid.shape[0])
    changes = 0
    for i in range(1, len(hs)):
        if abs(hs[i] - hs[i - 1]) >= 2:
            changes += 1
    return changes


def has_blind_gap(grid: np.ndarray):
    """
    Heuristic: a gap followed by a landing where the landing 'edge' isn't visible near jump point.
    Very rough: detect pattern of gap then sudden solid after >=2 gap run.
    """
    lg = longest_gap_run(grid)
    if lg >= 3:
        return True
    # Another pattern: gap col then immediate tall wall
    h = grid.shape[0]
    for c in range(grid.shape[1] - 1):
        if bottom_is_gap(grid, c) and is_solid(grid[h - 1, c + 1]):
            # if next column has a tall obstacle above bottom too
            if col_solid_height(grid, c + 1) <= h - 4:
                return True
    return False


def has_jump_streak(grid: np.ndarray):
    """
    Heuristic: multiple small gaps / discontinuities in short window suggests consecutive jumps.
    """
    # Count transitions solid<->gap on bottom
    r = grid.shape[0] - 1
    trans = 0
    last = is_solid(grid[r, 0])
    for c in range(1, grid.shape[1]):
        cur = is_solid(grid[r, c])
        if cur != last:
            trans += 1
        last = cur
    return trans >= 4  # threshold tuned for "streaky" areas


# ---------------------------
# Rule-based label assignment
# ---------------------------
def assign_label(grid: np.ndarray):
    """
    Priority-based label assignment.
    We pick ONE label (single-label Phase 1).
    """
    enemies = count_enemies(grid)
    gaps = count_gaps_on_bottom(grid)
    gap_run = longest_gap_run(grid)
    min_pw = min_platform_width(grid)
    height_changes = count_vertical_structure_changes(grid)

    # 1) Blind / dangerous gaps
    if has_blind_gap(grid):
        return 6  # GAP_BLIND

    # 2) Very large gap implies max jump arc
    if gap_run >= 5:
        return 4  # JUMP_ARC_MAX

    # 3) Jump streak zones (many transitions)
    if has_jump_streak(grid) and gaps >= 2:
        return 5  # JUMP_STREAK

    # 4) "Fail-rate" proxies: many enemies + complex terrain
    # (Phase 1 uses heuristic proxies; Phase 2 can replace with MCTS success rate.)
    if enemies >= 4 and height_changes >= 4:
        return 8  # FAIL_50
    if enemies >= 3 and height_changes >= 3:
        return 7  # FAIL_30

    # 5) Precision: narrow platforms / tight landings
    if 0 < min_pw <= 2:
        return 2  # PREC_HI
    if min_pw >= 6 and gap_run <= 1 and enemies <= 1:
        return 3  # PREC_LO

    # 6) Rhythm / flow
    if enemies <= 1 and gaps <= 1 and height_changes <= 1:
        return 0  # FLOW_FAST
    if enemies >= 2 or height_changes >= 3:
        return 1  # RHYTHM_STUTTER

    return 9  # NORMAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="./Mario-GPT2-700-context-length/behavior_labels.pt")
    ap.add_argument("--level_string", type=str, default=None, help="Optional: path to .txt or raw string")
    ap.add_argument("--context_len", type=int, default=700)
    ap.add_argument("--height", type=int, default=14)
    ap.add_argument("--sample_all_indices", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None, help="Optional: limit samples for quick test")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Build dataset (tokenizer is trained/loaded inside MarioDataset if not provided)
    dataset = MarioDataset(
        tokenizer=None,
        level_string=args.level_string,
        context_len=args.context_len,
        height=args.height,
        remove_start_end_tokens=False,
        sample_all_indices=args.sample_all_indices,
        behavior_labels_path=None,  # we are generating it
    )

    n = len(dataset)
    if args.max_samples is not None:
        n = min(n, int(args.max_samples))

    print(f"[generate] dataset length = {len(dataset)} (using n={n})")
    labels = torch.empty((n,), dtype=torch.long)

    # Generate labels
    for i in range(n):
        item = dataset[i]
        tokens = item[0]
        grid = tokens_to_grid(tokens, dataset.tokenizer, height=args.height)
        labels[i] = assign_label(grid)

    # Stats
    cnt = Counter(labels.tolist())
    print("[generate] label distribution:")
    for k in sorted(BEHAVIORS.keys()):
        name = BEHAVIORS[k]
        print(f"  {k:2d} {name:14s} : {cnt.get(k, 0)}")

    print(f"[generate] saving to: {args.output}")
    torch.save(labels, args.output)

    # Quick sanity
    print("[generate] saved tensor:")
    print("  shape:", tuple(labels.shape))
    print("  dtype:", labels.dtype)
    print("  min/max:", int(labels.min()), int(labels.max()))
    print("  example[0:10]:", labels[:10].tolist())


if __name__ == "__main__":
    main()
