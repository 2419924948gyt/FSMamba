# FSMamba

Code release for the paper:

**FSMamba: A Frequency-Enhanced Spatial-Temporal Mamba Network for Local-Global Remote Sensing Image Change Detection**

Paper link: [IEEE Xplore](https://ieeexplore.ieee.org/document/11515190/)

## Environment

Ubuntu with a CUDA GPU is recommended. The conda environment name is `rscd`.

```bash
conda create -n rscd python=3.8 -y
conda activate rscd

# Example for CUDA 11.8. Change this command if your CUDA version is different.
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install causal-conv1d==1.2.0.post1
pip install mamba-ssm==1.2.0.post1
```

If `causal-conv1d` or `mamba-ssm` cannot be installed from pip, install wheel files matching your Ubuntu, CUDA, PyTorch, and Python versions.

## Data

Put datasets under `data/` or edit `data_root` in `configs/_base_/`.

Default CLCD path:

```text
data/CLCD/
```

## Training

```bash
python train.py -c configs/FSMamba.py
```

Training logs and checkpoints will be saved under:

```text
work_dirs/clcd_epoch100/FSMamba/
```

## Testing

```bash
python test.py \
  -c configs/FSMamba.py \
  --ckpt work_dirs/clcd_epoch100/FSMamba/version_x/ckpts/test/test_change_f1/best.ckpt \
  --output_dir work_dirs/eval_fsmamba_clcd
```

Evaluation metrics are printed in the terminal. RGB masks are saved to:

```text
work_dirs/eval_fsmamba_clcd/mask_rgb/
```

## Citation

```bibtex
@article{guo2026fsmamba,
  title={FSMamba: A Frequency-Enhanced Spatial-Temporal Mamba Network for Local-Global Remote Sensing Image Change Detection},
  author={Yingting Guo and Yan Xu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```
