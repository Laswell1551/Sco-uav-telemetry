import json

import numpy as np

from run_tmc_v16_trace_baseline_expansion import (
    CONTROLLED_SCHEMA_VERSION,
    _selected_observation,
    load_frozen_choices,
    run_de_trace_policy,
    run_dts_trace_policy,
    summarize_episodes,
    trace_posterior_normal_bank,
)


def small_physical(arms=4, length=28):
    time = np.linspace(0.0, 1.0, length)
    q = np.stack(
        [
            np.clip(
                0.05 + 0.15 * arm + 0.35 * time,
                0.01,
                1.0,
            )
            for arm in range(arms)
        ]
    )
    r = np.full_like(q, 0.05)
    return np.stack([q, r], axis=-1)


def test_selected_observation_masks_unselected_entries():
    current = np.arange(8, dtype=float).reshape(1, 4, 2)
    selected = np.array([[True, False, False, True]])
    observed = _selected_observation(current, selected)
    assert np.array_equal(observed[selected], current[selected])
    assert np.count_nonzero(observed[~selected]) == 0


def test_trace_policies_are_finite_and_use_exact_budget():
    physical = small_physical()
    horizon = physical.shape[1] - 8
    de = run_de_trace_policy(
        physical, alpha=1.0, budget=1, n0=8
    )
    normals = np.zeros((horizon, 1, 4, 2), dtype=float)
    dts = run_dts_trace_policy(
        physical,
        gamma=0.98,
        posterior_standard_normals=normals,
        budget=1,
        n0=8,
    )
    for result in (de, dts):
        assert np.isfinite(result["cost"])
        assert np.isfinite(result["rank_loss"])
        assert result["selected_observation_blocks"] == horizon
        assert result["max_gap"] >= 0
        assert result["runtime_seconds"] >= 0
    assert 0.0 <= de["forced_slot_fraction"] <= 1.0
    assert 0.0 <= dts["posterior_clip_fraction"] <= 1.0


def test_normal_bank_is_reproducible_and_dataset_separated():
    first = trace_posterior_normal_bank(
        "uzh_fpv", 410001, horizon=5, arms=4
    )
    second = trace_posterior_normal_bank(
        "uzh_fpv", 410001, horizon=5, arms=4
    )
    other = trace_posterior_normal_bank(
        "m3ed_falcon", 410001, horizon=5, arms=4
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_frozen_choices_require_disjoint_pilot_contract(tmp_path):
    path = tmp_path / "controlled.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": CONTROLLED_SCHEMA_VERSION,
                "protocol": {
                    "formal_data_not_used_for_tuning": True
                },
                "pilot": {
                    "choices": {
                        "de_alpha": 1.0,
                        "dts_gamma": 0.98,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    choices, _ = load_frozen_choices(path)
    assert choices == {"de_alpha": 1.0, "dts_gamma": 0.98}


def test_episode_summary_contains_paired_sco_contrast():
    episodes = []
    for index in range(4):
        sco_excess = 5.0 + index
        episodes.append(
            {
                "references": {
                    "sco_reset_ucb": {
                        "cost": 100.0 + index,
                        "excess_pct": sco_excess,
                        "rank_loss": 2.0,
                        "max_gap": 4,
                        "alarms_per_10k_arm_slots": 1.0,
                    }
                },
                "methods": {
                    "de_cd_whittle_cv": {
                        "cost": 103.0 + index,
                        "excess_pct": sco_excess + 2.0,
                        "rank_loss": 3.0,
                        "max_gap": 5,
                        "alarms_per_10k_arm_slots": 1.2,
                        "forced_slot_fraction": 0.1,
                        "exploration_blocks": 2,
                    },
                    "dts_whittle_cv": {
                        "cost": 102.0 + index,
                        "excess_pct": sco_excess + 1.0,
                        "rank_loss": 2.5,
                        "max_gap": 6,
                        "posterior_clip_fraction": 0.02,
                        "final_effective_weight_mean": 5.0,
                    },
                },
            }
        )
    summary = summarize_episodes(
        "uzh_fpv", episodes, bootstrap_replicates=1000
    )
    de_diff = summary["paired_method_minus_sco"][
        "de_cd_whittle_cv"
    ]["mean_ci95"]
    dts_diff = summary["paired_method_minus_sco"][
        "dts_whittle_cv"
    ]["mean_ci95"]
    assert de_diff[0] == 2.0
    assert dts_diff[0] == 1.0
    assert summary["bootstrap_replicates"] == 1000
