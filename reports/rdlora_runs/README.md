# RD-LoRA Stage D Artifacts

This directory documents the hardened Stage D artifact contracts used by the runtime checks and tests.

Probe outputs:
- `run_provenance.json`
- `probe_summary.json`
- `cell_utility.csv`
- `cell_utility.json`

Surrogate outputs:
- `surrogate_summary.json`
- `surrogate_model.pkl`

Allocation outputs:
- `allocation_summary.json`
- `uniform.json`
- `layer_only.json`
- `timestep_only.json`
- `proposed.json`

Train outputs:
- `run_provenance.json`
- `train_summary.json`
- `metrics.json`
- `checkpoint_info.json`

Evaluation outputs:
- `evaluation_summary.json`
- `baseline_metrics.csv`
- `baseline_metrics.json`

Gate outputs:
- `gate_memo.md`
- `gate_memo.json`

Runtime policy:
- `eval_rdlora.py` refuses runs without `real_gpu` provenance.
- `write_rdlora_gate_memo.py` emits `INVALID` instead of collapsing invalid inputs into `VALID_NO_GO`.
- Mock and smoke source paths are rejected in eval and gate flows.
