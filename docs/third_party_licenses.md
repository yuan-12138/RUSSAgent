# Third-party software and models

RUSSAgent code is released under MIT. Third-party components retain their own licenses.

| Component | Integration | License / source |
|-----------|-------------|------------------|
| CLIFF | Vendored integration code; weights excluded | CLIFF MIT license included with the source |
| SKEL | Cloned by `scripts/setup.sh`; model files excluded | SKEL research license; registration required |
| SMPL / SMPL-H / MANO | Not distributed | Max Planck model licenses; registration required |
| easy_handeye | Cloned by `scripts/setup.sh` | Upstream repository license |
| Azure Kinect ROS Driver | Cloned by `scripts/setup.sh` | Microsoft upstream license |
| pytorch-yolo-v3 | Cloned by `scripts/setup.sh`; weights excluded | Upstream repository terms |
| iiwa_stack | Vendored ROS packages | BSD (declared in package manifests) |

Do not commit model files under `workspace/models/**/data/` or `cliff_repo/data/`. Users must obtain them from the original providers.
