# Humanoid Arm Workspace Sweep -- Results Summary

## Experiment Metadata

- **Date**: 2026-04-29T08:20:02.891178+00:00
- **Robot model**: /home/yaashia/cerg/examples/h12/models/scene_handless.xml
- **Standing policy**: /home/yaashia/h12_adaptive_policy/data/rma_hand
- **MuJoCo version**: 3.5.0
- **Sim timestep**: 0.002s
- **Episode duration**: 4.0s
- **Num trials**: 1000
- **Fall height threshold**: 50% of initial
- **Fall orientation threshold**: 60.0deg
- **Initial standing height**: 1.0300m
- **Execution time**: 131.2s (2.2 min)
- **Seed**: 42

## Default PD Gains (CERG config)

| Joint | Default KP | Default KD |
|-------|-----------|-----------|
| left_shoulder_pitch | 30.0 | 8.0 |
| left_shoulder_roll | 30.0 | 8.0 |
| left_shoulder_yaw | 15.0 | 4.0 |
| left_elbow | 15.0 | 4.0 |
| left_wrist_roll | 10.0 | 3.0 |
| left_wrist_pitch | 10.0 | 3.0 |
| left_wrist_yaw | 10.0 | 3.0 |
| right_shoulder_pitch | 30.0 | 8.0 |
| right_shoulder_roll | 30.0 | 8.0 |
| right_shoulder_yaw | 15.0 | 4.0 |
| right_elbow | 15.0 | 4.0 |
| right_wrist_roll | 10.0 | 3.0 |
| right_wrist_pitch | 10.0 | 3.0 |
| right_wrist_yaw | 10.0 | 3.0 |

## Overall Statistics

- **Total trials**: 1000
- **Falls**: 733 (73.3%)
- **Safe**: 267 (26.7%)

## Breakdown by Arm Mode

| Mode | Trials | Falls | Safe | Fall Rate |
|------|--------|-------|------|-----------|
| left_only | 334 | 235 | 99 | 70.4% |
| right_only | 333 | 211 | 122 | 63.4% |
| both | 333 | 287 | 46 | 86.2% |

## Per-Joint Analysis

### Safe Target Angle Ranges

| Joint | Safe Min (rad) | Safe Max (rad) | Joint Range | % of Range Usable |
|-------|---------------|---------------|-------------|-------------------|
| left_shoulder_pitch | -3.066 | 1.510 | [-3.14, 1.57] | 97% |
| left_shoulder_roll | -0.367 | 3.059 | [-0.38, 3.40] | 91% |
| left_shoulder_yaw | -2.656 | 3.001 | [-2.66, 3.01] | 100% |
| left_elbow | -0.947 | 3.171 | [-0.95, 3.18] | 100% |
| left_wrist_roll | -2.998 | 2.748 | [-3.01, 2.75] | 100% |
| left_wrist_pitch | -0.456 | 0.462 | [-0.46, 0.46] | 99% |
| left_wrist_yaw | -1.241 | 1.264 | [-1.27, 1.27] | 99% |
| right_shoulder_pitch | -3.139 | 1.402 | [-3.14, 1.57] | 96% |
| right_shoulder_roll | -3.303 | 0.378 | [-3.40, 0.38] | 97% |
| right_shoulder_yaw | -2.985 | 2.658 | [-3.01, 2.66] | 100% |
| right_elbow | -0.895 | 3.154 | [-0.95, 3.18] | 98% |
| right_wrist_roll | -2.571 | 3.008 | [-2.75, 3.01] | 97% |
| right_wrist_pitch | -0.462 | 0.462 | [-0.46, 0.46] | 100% |
| right_wrist_yaw | -1.270 | 1.265 | [-1.27, 1.27] | 100% |

### Gain Sensitivity

- **left_shoulder_pitch**: High KP fall rate=75.1%, Low KP=84.2% -> tolerant
- **left_shoulder_roll**: High KP fall rate=74.3%, Low KP=81.5% -> tolerant
- **left_shoulder_yaw**: High KP fall rate=76.5%, Low KP=83.4% -> tolerant
- **left_elbow**: High KP fall rate=75.0%, Low KP=77.6% -> tolerant
- **left_wrist_roll**: High KP fall rate=75.8%, Low KP=83.2% -> tolerant
- **left_wrist_pitch**: High KP fall rate=79.9%, Low KP=80.1% -> tolerant
- **left_wrist_yaw**: High KP fall rate=71.0%, Low KP=83.2% -> SENSITIVE
- **right_shoulder_pitch**: High KP fall rate=68.2%, Low KP=81.0% -> SENSITIVE
- **right_shoulder_roll**: High KP fall rate=72.7%, Low KP=79.0% -> tolerant
- **right_shoulder_yaw**: High KP fall rate=75.8%, Low KP=78.7% -> tolerant
- **right_elbow**: High KP fall rate=72.3%, Low KP=73.8% -> tolerant
- **right_wrist_roll**: High KP fall rate=75.4%, Low KP=79.2% -> tolerant
- **right_wrist_pitch**: High KP fall rate=76.9%, Low KP=74.3% -> tolerant
- **right_wrist_yaw**: High KP fall rate=77.3%, Low KP=74.2% -> tolerant

## Key Findings

### Joint Sensitivity
- Both-arm mode falls significantly more (86.2% vs 66.9% single-arm)
- Fast falls (<0.5s): 0 (0% of falls)
- Slow falls (>2.0s): 130 (18% of falls)
- Falls are predominantly slow drifts, suggesting the policy struggles to compensate over time

### CoM Deviation Threshold
- 95th percentile safe CoM deviation: 0.1700m
- This suggests a CERG soft constraint boundary around 0.170m horizontal CoM shift

### Most Extreme Safe Configurations
- Trial 255: mode=right_only, CoM_dev=0.5682m, height=0.7864m
- Trial 121: mode=both, CoM_dev=0.5193m, height=0.9073m
- Trial 764: mode=right_only, CoM_dev=0.5032m, height=0.8388m

## Implications for CERG Constraint Design

- The CoM deviation threshold maps directly to the energy/safety soft constraint boundary
- Joints with narrow safe ranges need tighter CERG constraints
- Both-arm scenarios may need more conservative limits than single-arm
- Fast-fall trials identify hard boundaries; slow-fall trials identify soft boundaries

## Created Files

| File | Description |
|------|-------------|
| `results/workspace_trials.csv` | Per-trial results (1000 rows) |
| `results/experiment_config.json` | Experiment metadata and config |
| `results/trajectories.npz` | Time-series trajectories (subsampled) |
| `results/workspace_summary.md` | This summary report |
| `results/plots/per_joint_safe_range_single.png/pdf` | Plot 1a |
| `results/plots/per_joint_safe_range_both.png/pdf` | Plot 1b |
| `results/plots/kpkd_sensitivity_single.png/pdf` | Plot 2a |
| `results/plots/kpkd_sensitivity_both.png/pdf` | Plot 2b |
| `results/plots/arm_mode_comparison.png/pdf` | Plot 3 |
| `results/plots/time_to_fall_hist.png/pdf` | Plot 4 |
| `results/plots/com_deviation_scatter.png/pdf` | Plot 5 |
| `results/plots/joint_pair_scatter.png/pdf` | Plot 6 |
| `results/plots/dashboard.png/pdf` | Plot 7 |
