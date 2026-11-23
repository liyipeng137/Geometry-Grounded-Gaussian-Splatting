from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from train_real_geometry import cadence_windows, refinement_policy
from utils.real_geometry_topology import RealGeometryTopology, TopologyConfig


def test_accelerated_growth_preserves_reference_prune_windows():
    cfg = TopologyConfig()
    growth, prune = cadence_windows(20000, cfg)
    steps = [
        i
        for i in range(cfg.first, 20001, cfg.interval)
        if refinement_policy(i, 20000)[2]
    ]
    assert prune == list(zip(steps[::2], steps[1::2]))
    assert len(prune) == 35 and len(growth) == 73
    assert growth[:6] == prune[:6]
    assert growth[6] == (3100, 3200) and growth[-9] == (14700, 14800)
    assert growth[-8] == (15000, 15200) and growth[-1] == (17800, 18000)
    assert all(b < c for (_, b), (c, _) in zip(growth, growth[1:]))


def test_cadence_draws_and_pending_prune_survive_a_growth_commit():
    def make():
        t = object.__new__(RealGeometryTopology)
        t.rng = np.random.default_rng(0)
        t.cams = [
            SimpleNamespace(nearest_id=[(i + 1) % 6, (i + 2) % 6]) for i in range(6)
        ]
        t.cfg, t.remaining = TopologyConfig(views=3), []
        t.meta = {"birth": torch.zeros(3, dtype=torch.long)}
        t.counts = Counter()
        return t

    t, reference = make(), make()
    t.configure_cadence([(1, 2), (3, 4)], [(2, 4)])
    assert t.panels == [reference.choose_pairs(), reference.choose_pairs()]

    def measure(step, pairs):
        flag = torch.ones(3, dtype=torch.bool)
        return flag.float(), flag, flag, flag, {}

    t._measure = measure
    commits = []

    def commit(step, growth, pruning):
        commits.append((step, growth, pruning))
        if step == 2:
            # One survivor ordering and a newly inserted row; mirror the real transaction.
            for key in ("growth_first", "prune_first"):
                t.meta[key] = torch.cat((t.meta[key][[2, 0]], torch.tensor([False])))

    t._commit = commit
    for i in range(1, 5):
        t._scheduled_step(i)
    assert [c[0] for c in commits] == [2, 4]
    assert commits[0][2] is None and commits[1][2][0].tolist() == [True, True, False]
    assert commits[1][2][2] == t.panels[0]
