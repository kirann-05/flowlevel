# -*- coding: utf-8 -*-
"""
bfs_checker.py - BFS-based Sokoban level analysis.

difficulty_metrics(level) returns:
  solution_len : minimum moves to solve (-1 = unsolvable)
  box_dist     : sum of box-to-goal Manhattan distances
  n_boxes      : number of boxes
  solvable     : 1.0 if solvable, 0.0 if not
"""

from collections import deque
from typing import Dict
import numpy as np

EMPTY=0; SOLID=1; PLAYER=2; CRATE=3
TARGET=4; CRATE_ON_TARGET=5; PLAYER_ON_TARGET=6
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]


def _parse(level):
    player, boxes, goals = None, [], []
    for r in range(level.shape[0]):
        for c in range(level.shape[1]):
            t = int(level[r,c])
            if t in (PLAYER, PLAYER_ON_TARGET):
                player = (r,c)
                if t == PLAYER_ON_TARGET: goals.append((r,c))
            elif t == CRATE:           boxes.append((r,c))
            elif t == TARGET:          goals.append((r,c))
            elif t == CRATE_ON_TARGET: boxes.append((r,c)); goals.append((r,c))
    return player, boxes, goals


def solution_length(level, max_steps=1000):
    player, boxes, goals = _parse(level)
    if not player or not boxes or not goals or len(boxes) != len(goals):
        return -1
    goal_set = frozenset(goals)
    solid    = set(zip(*np.where(level == SOLID)))
    init     = (player, frozenset(boxes))
    visited  = {init}
    queue    = __import__('collections').deque([(init, 0)])
    while queue:
        (pos, box_set), steps = queue.popleft()
        if steps >= max_steps: continue
        if box_set == goal_set: return steps
        r, c = pos
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if (nr,nc) in solid: continue
            new_boxes = box_set
            if (nr,nc) in box_set:
                br, bc = nr+dr, nc+dc
                if (br,bc) in solid or (br,bc) in box_set: continue
                new_boxes = (box_set - {(nr,nc)}) | {(br,bc)}
            ns = ((nr,nc), new_boxes)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, steps+1))
    return -1


def is_solvable(level):
    return solution_length(level) >= 0


def box_distances(level):
    _, boxes, goals = _parse(level)
    if not boxes or not goals: return 0.0
    return float(sum(min(abs(br-gr)+abs(bc-gc) for gr,gc in goals) for br,bc in boxes))


def difficulty_metrics(level) -> Dict[str, float]:
    sol = solution_length(level)
    return {
        'solution_len': float(sol),
        'box_dist':     box_distances(level),
        'n_boxes':      float(len(_parse(level)[1])),
        'solvable':     1.0 if sol >= 0 else 0.0,
    }


if __name__ == '__main__':
    import sys
    tests = [
        {'name': 'Trivial 1-move',
         'level': np.array([[1,1,1,1,1],[1,2,3,4,1],[1,0,0,0,1],[1,1,1,1,1]]),
         'solvable': True, 'min_len': 1},
        {'name': 'Two-move solve',
         'level': np.array([[1,1,1,1,1,1],[1,2,0,3,4,1],[1,0,0,0,0,1],[1,1,1,1,1,1]]),
         'solvable': True, 'min_len': 2},
        {'name': 'Unsolvable corner',
         'level': np.array([[1,1,1,1,1],[1,3,1,0,1],[1,1,0,4,1],[1,2,0,0,1],[1,1,1,1,1]]),
         'solvable': False, 'min_len': -1},
        {'name': 'Already solved',
         'level': np.array([[1,1,1,1,1],[1,2,0,5,1],[1,0,0,0,1],[1,1,1,1,1]]),
         'solvable': True, 'min_len': 0},
    ]
    passed = 0
    for t in tests:
        sol = is_solvable(t['level'])
        ln  = solution_length(t['level'])
        ok  = (sol == t['solvable']) and (ln >= t['min_len'])
        print(f"  [{'PASS' if ok else 'FAIL'}]  {t['name']}  solvable={sol}  len={ln}")
        if ok: passed += 1
    print(f"\n  {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
