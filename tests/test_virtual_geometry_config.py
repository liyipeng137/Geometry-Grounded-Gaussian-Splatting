"""Stage boundaries and mistakes that would silently change the production run."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.virtual_geometry_config import DEFAULTS, load_joint_config, virtual_schedule


def test_default_schedule_keeps_real_20k_and_adds_6625_virtual_views():
    samples = [i for i in range(1, 20001) if virtual_schedule(i, DEFAULTS)[0]]
    assert len(samples) == 6625
    assert samples[0] == 3504 and samples[-1] == 20000
    assert virtual_schedule(3500, DEFAULTS) == (False, 0., "real_anchor")
    assert virtual_schedule(5000, DEFAULTS) == (True, 1., "virtual_ramp")
    assert virtual_schedule(15000, DEFAULTS) == (True, 1., "joint_geometry")
    assert virtual_schedule(15004, DEFAULTS) == (True, 1., "fixed_topology")


def test_manifest_relative_to_config_and_weights_merge(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"teacher_manifest": "teacher/manifest.json", "loss_weights": {"depth": .1}}))
    config = load_joint_config(p)
    assert config["teacher_manifest"] == str(tmp_path / "teacher/manifest.json")
    assert config["loss_weights"]["depth"] == .1
    assert config["loss_weights"]["normal_render"] == .05
    assert DEFAULTS["loss_weights"]["depth"] == .2


@pytest.mark.parametrize("change", [
    {"typo": 1}, {"loss_weights": {"depth": -1}}, {"loss_weights": {"unknown": 1}},
    {"seed_until": 16000}, {"last_opacity_reset": 6000}, {"warm_interval": 0},
    {"geometry_reserve": 1}, {"depth_search_expansions": 0},
])
def test_invalid_joint_config_fails_before_training(tmp_path, change):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"teacher_manifest": "cache.json", **change}))
    with pytest.raises(ValueError):
        load_joint_config(p)
