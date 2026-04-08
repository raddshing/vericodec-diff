from __future__ import annotations

import csv
import json
import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.allocator import solve_multiple_choice_knapsack, validate_allocation_manifest


PROBE_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora_surrogate.py"
SOLVE_SCRIPT = REPO_ROOT / "scripts" / "solve_rdlora_allocation.py"


def _brute_force_choice_groups(choice_groups: list[dict[str, object]], budget: int) -> dict[str, object]:
    best_result: dict[str, object] | None = None
    ordered_groups = sorted(choice_groups, key=lambda item: str(item["group_id"]))
    option_lists = [list(group["options"]) for group in ordered_groups]
    for selection_tuple in itertools.product(*option_lists):
        used_budget = sum(int(option["cost"]) for option in selection_tuple)
        if used_budget > budget:
            continue
        total_utility = round(sum(float(option["utility"]) for option in selection_tuple), 6)
        signature = tuple(int(option["candidate_rank"]) for option in selection_tuple)
        candidate = {
            "used_budget": used_budget,
            "total_predicted_utility": total_utility,
            "signature": signature,
        }
        if best_result is None:
            best_result = candidate
            continue
        if total_utility > float(best_result["total_predicted_utility"]):
            best_result = candidate
            continue
        if total_utility == float(best_result["total_predicted_utility"]) and used_budget < int(best_result["used_budget"]):
            best_result = candidate
            continue
        if (
            total_utility == float(best_result["total_predicted_utility"])
            and used_budget == int(best_result["used_budget"])
            and signature < tuple(best_result["signature"])
        ):
            best_result = candidate
    if best_result is None:
        raise AssertionError("Synthetic brute-force instance has no feasible solution")
    return best_result


class RdLoraAllocatorTests(unittest.TestCase):
    def test_exact_solver_matches_bruteforce_on_tiny_instance(self) -> None:
        choice_groups = [
            {
                "group_id": "cell_a",
                "is_attention_cell": True,
                "options": [
                    {"cell_id": "cell_a", "layer_group_id": "g0", "timestep_band_id": "t0", "candidate_rank": 0, "cost": 0, "utility": 0.0, "rank_fraction_of_max": 0.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_a", "layer_group_id": "g0", "timestep_band_id": "t0", "candidate_rank": 2, "cost": 2, "utility": 6.0, "rank_fraction_of_max": 0.5, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_a", "layer_group_id": "g0", "timestep_band_id": "t0", "candidate_rank": 4, "cost": 4, "utility": 9.0, "rank_fraction_of_max": 1.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                ],
            },
            {
                "group_id": "cell_b",
                "is_attention_cell": True,
                "options": [
                    {"cell_id": "cell_b", "layer_group_id": "g1", "timestep_band_id": "t0", "candidate_rank": 0, "cost": 0, "utility": 0.0, "rank_fraction_of_max": 0.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_b", "layer_group_id": "g1", "timestep_band_id": "t0", "candidate_rank": 2, "cost": 2, "utility": 5.0, "rank_fraction_of_max": 0.5, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_b", "layer_group_id": "g1", "timestep_band_id": "t0", "candidate_rank": 4, "cost": 4, "utility": 8.0, "rank_fraction_of_max": 1.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                ],
            },
            {
                "group_id": "cell_c",
                "is_attention_cell": True,
                "options": [
                    {"cell_id": "cell_c", "layer_group_id": "g2", "timestep_band_id": "t0", "candidate_rank": 0, "cost": 0, "utility": 0.0, "rank_fraction_of_max": 0.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_c", "layer_group_id": "g2", "timestep_band_id": "t0", "candidate_rank": 1, "cost": 1, "utility": 3.0, "rank_fraction_of_max": 0.25, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_c", "layer_group_id": "g2", "timestep_band_id": "t0", "candidate_rank": 2, "cost": 2, "utility": 4.0, "rank_fraction_of_max": 0.5, "layer_count": 1, "step_count": 1, "event_count": 1},
                ],
            },
        ]
        budget = 5

        expected = _brute_force_choice_groups(choice_groups, budget)
        result = solve_multiple_choice_knapsack(
            choice_groups,
            budget=budget,
            utility_scale=1000,
            max_exact_state_count=100000,
        )

        self.assertEqual(result["solver_mode"], "exact_dynamic_programming")
        self.assertEqual(result["used_budget"], expected["used_budget"])
        self.assertEqual(result["total_predicted_utility"], expected["total_predicted_utility"])
        self.assertEqual(
            tuple(int(item["candidate_rank"]) for item in result["selections"]),
            tuple(expected["signature"]),
        )

    def test_solver_tie_break_is_deterministic(self) -> None:
        choice_groups = [
            {
                "group_id": "cell_a",
                "is_attention_cell": True,
                "options": [
                    {"cell_id": "cell_a", "layer_group_id": "g0", "timestep_band_id": "t0", "candidate_rank": 0, "cost": 0, "utility": 0.0, "rank_fraction_of_max": 0.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_a", "layer_group_id": "g0", "timestep_band_id": "t0", "candidate_rank": 1, "cost": 1, "utility": 1.0, "rank_fraction_of_max": 1.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                ],
            },
            {
                "group_id": "cell_b",
                "is_attention_cell": True,
                "options": [
                    {"cell_id": "cell_b", "layer_group_id": "g1", "timestep_band_id": "t0", "candidate_rank": 0, "cost": 0, "utility": 0.0, "rank_fraction_of_max": 0.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                    {"cell_id": "cell_b", "layer_group_id": "g1", "timestep_band_id": "t0", "candidate_rank": 1, "cost": 1, "utility": 1.0, "rank_fraction_of_max": 1.0, "layer_count": 1, "step_count": 1, "event_count": 1},
                ],
            },
        ]

        first = solve_multiple_choice_knapsack(choice_groups, budget=1, utility_scale=1000)
        second = solve_multiple_choice_knapsack(choice_groups, budget=1, utility_scale=1000)

        self.assertEqual(first, second)
        self.assertEqual(
            [int(item["candidate_rank"]) for item in first["selections"]],
            [0, 1],
        )

    def test_mocked_probe_smoke_train_and_solve_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            temp_root = Path(tmpdir)
            probe_root = temp_root / "probe_outputs"
            allocator_root = temp_root / "allocator_outputs"
            probe_run_dir = probe_root / "probe_smoke"
            run_name = "stagec_smoke"

            subprocess.run(
                [
                    sys.executable,
                    str(PROBE_SCRIPT),
                    "--config",
                    "configs/rdlora_probe.yaml",
                    "--run-name",
                    "probe_smoke",
                    "--set",
                    f"paths.output_root={json.dumps(str(probe_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                    "--set",
                    "backend.mock.sample_count=2",
                    "--set",
                    "backend.mock.vector_size=4",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            train_completed = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--config",
                    "configs/rdlora_allocator.yaml",
                    "--run-name",
                    run_name,
                    "--set",
                    f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                    "--set",
                    f"paths.output_root={json.dumps(str(allocator_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("preflight_ok=1", train_completed.stdout)

            solve_command = [
                sys.executable,
                str(SOLVE_SCRIPT),
                "--config",
                "configs/rdlora_allocator.yaml",
                "--run-name",
                run_name,
                "--set",
                f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                "--set",
                f"paths.output_root={json.dumps(str(allocator_root))}",
                "--set",
                "allocation.budget=40",
                "--set",
                "preflight.require_torch_cuda=false",
                "--set",
                "preflight.required_diffusers_version=null",
            ]
            first_solve = subprocess.run(
                solve_command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_path = allocator_root / run_name / "allocation" / "allocation_manifest.json"
            first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            second_solve = subprocess.run(
                solve_command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertIn("preflight_ok=1", first_solve.stdout)
            self.assertIn("preflight_ok=1", second_solve.stdout)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["solver"]["mode"], "exact_dynamic_programming")
            self.assertLessEqual(first_manifest["budget"]["used"], first_manifest["budget"]["available"])
            self.assertEqual(len(first_manifest["selections"]), 24)

            validation = validate_allocation_manifest(manifest_path)
            self.assertTrue(validation["ok"])

            with (allocator_root / run_name / "allocation" / "allocation_choices.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)


if __name__ == "__main__":
    unittest.main()
