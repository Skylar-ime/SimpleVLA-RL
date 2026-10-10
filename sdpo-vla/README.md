# SDPO-VLA experiments

The pilot implementations, research notes, and audited comparison results are kept here. Each pilot version has its own `doc/` directory for experiment reports.

- `sdpo_result_feedback_pilot/`: initial PandaPush result-feedback pilot.
- `sdpo_result_feedback_pilot_v5/`, `_v6/`, `_v7/`, `_v11/`: successive protocols and evaluations.
- `sdpo_vla_research/`: literature snapshots and the [VLA research report](sdpo_vla_research/doc/SDPO_VLA_全链条汇报内容.md).

Downloaded model weights and full run artifacts remain local and are ignored by Git. Small comparison, report, summary, and audit JSON files are versioned to support the written results. Scripts use local Conda and GPU resources; set those environment paths for the target machine before running them.

Local storage cleanup on 2026-10-10 removed 1,729 intermediate `training_state.pt` files (116.47 GiB) from completed v7 and v11 runs. Each completed run retains its final optimizer state and adapter; incomplete runs were untouched. Exact resume from a removed intermediate update is no longer possible. Base, v5, and v6 optimizer states remain because their audit scripts read every update state.
