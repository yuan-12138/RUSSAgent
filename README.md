# RUSSAgent

**From Scanning Guidelines to Action: A Robotic Ultrasound Agent with LLM-Based Reasoning**

[![Paper](https://img.shields.io/badge/arXiv-2603.14393-b31b1b)](https://arxiv.org/abs/2603.14393)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://youtu.be/pfMOc4e2IGA)

Guideline-driven LLM agent for autonomous robotic ultrasound on **gallbladder, kidney, and spine**, using CLIFF+SKEL trajectory planning with KUKA LBR IIWA and Azure Kinect.

**Platform:** Ubuntu 20.04 · ROS Noetic · Python 3.8 · CUDA 11.8

> **Research only.** This software is not a medical device. Validate robot motion in simulation and at low speed with an emergency stop before any human contact.

> **Scope:** Real-robot inference demo and agent framework. RL/SFT training code and model weights are not included.

## Quick start

```bash
git clone https://github.com/yuan-12138/RUSSAgent.git
cd RUSSAgent
bash scripts/setup.sh
```

Download the models listed below, complete [hand-eye calibration](docs/calibration.md), and then run:

```bash
bash scripts/start_russagent.sh
```

The launcher checks the model files, asks for the OpenAI-compatible API endpoint and model name, brings up MoveIt + Azure Kinect + hand-eye TF, and then starts the agent. Real-robot motion requires typing `START`.

`setup.sh` automatically installs the required source dependencies, creates the conda environment, and builds the ROS workspace.

## Download models

These five files are required and are not included in this repository:

| File | Put it here | Source |
|------|-------------|--------|
| CLIFF HRNet-48 | `workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/cliff_repo/data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt` | [CLIFF Drive](https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP) |
| YOLOv3 | `workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/cliff_repo/data/ckpt/yolov3.weights` | [Download](https://pjreddie.com/media/files/yolov3.weights) |
| SMPL mean parameters | `workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/cliff_repo/data/smpl_mean_params.npz` | [CLIFF Drive](https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP) |
| SMPL neutral model | `workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/cliff_repo/data/smpl/SMPL_NEUTRAL.pkl` | [SMPL](https://smpl.is.tue.mpg.de) (registration required) |
| SKEL male model | `workspace/models/SKEL/data/skel/skel_male.pkl` | [SKEL](https://skel.is.tue.mpg.de) (registration required) |

## Project structure

The handbook defines the scanning workflow, the API catalog defines the actions available to the LLM, and the tools ground those actions in perception and robot execution.

```text
RUSSAgent/
├── OrchestratorRUSS/
│   ├── scripts/
│   │   ├── run_agent.py
│   │   └── start_agent.sh
│   ├── launch/
│   └── config/
├── scripts/
│   ├── setup.sh
│   └── start_russagent.sh
└── workspace/
    ├── models/SKEL/
    └── src/
        ├── autonomous_scan_agent/
        │   ├── prompts/
        │   │   ├── russagent_handbook.txt
        │   │   └── russagent_api_catalog.json
        │   ├── scripts/
        │   │   └── rgbpair_skel_pipeline.py
        │   └── src/autonomous_scan_agent/
        │       ├── llm_client.py
        │       ├── llm_interface.py
        │       └── tools/
        │           ├── tool_0_operator_io.py
        │           ├── tool_1_trajectory_acquisition.py
        │           ├── tool_2_trajectory_projection.py
        │           ├── tool_3_contact_verification.py
        │           ├── tool_4_path_execution.py
        │           ├── tool_5_reset_to_capture_pose.py
        │           ├── tool_6_post_scan_adjustment.py
        │           └── cliff_repo/
        └── image_processing/
```

## Docs

- [Hardware](docs/hardware.md)
- [Calibration](docs/calibration.md)
- [Third-party licenses](docs/third_party_licenses.md)

## Citation

```bibtex
@article{bi2026russagent,
  title={From Scanning Guidelines to Action: A Robotic Ultrasound Agent with LLM-Based Reasoning},
  author={Bi, Yuan and Zhou, Yiping and others},
  journal={arXiv:2603.14393},
  year={2026}
}
```

## License

MIT for this repository's original code. Third-party: CLIFF (MIT), easy_handeye, iiwa_stack, Azure Kinect driver, SMPL/SKEL (Max Planck non-commercial — **not redistributed**).
