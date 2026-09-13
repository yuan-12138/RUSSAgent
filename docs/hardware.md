# Hardware Setup

Real-world experiments in the paper used:

| Component | Model |
|-----------|-------|
| Robot | KUKA LBR IIWA + MoveIt |
| RGB-D camera | Microsoft Azure Kinect (eye-in-hand) |
| Ultrasound | Siemens ACUSON Juniper, 5C1 probe |
| Compute | NVIDIA GPU (CUDA 11.8), RTX 4070 class |

This repository controls **robot motion + RGB-D perception**. Ultrasound image capture (Epiphan frame grabber) is **not** included in the open-source demo; contact/refinement uses operator feedback.

## Software

- Ubuntu 20.04
- ROS Noetic
- Azure Kinect SDK
- conda env `Russ_agent` (see `environment.yml`)

## Safety

Real robot execution is disabled by default. Only set `RUSSAGENT_ENABLE_ROBOT=1` after validating transforms and trajectories in simulation and at low speed, with workspace supervision and an accessible emergency stop. The supplied calibration pose example is intentionally empty.
