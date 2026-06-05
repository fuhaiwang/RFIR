## Radio-Frequency Inverse Rendering for Wireless Environment Digital Twins


## Install

```bash
git clone https://github.com/fuhaiwang/RFIR.git
cd RFIR

conda create -n RFIR python=3.8 -y
conda activate RFIR

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit

pip install \
  matplotlib==3.7.2 \
  tqdm==4.64.1 \
  pillow==10.4.0 \
  pip==24.2

pip install \
  dearpygui \
  imageio \
  opencv-python \
  plyfile \
  pyexr \
  scipy \
  pybind11 \
  tensorboard
```

## Install torch_scatter, Kornia & nvdiffrast
```bash
pip install torch_scatter \
  --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu118/
```

```bash
pip install kornia

git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```
## A modified gaussian splatting & simple-knn 
```bash
pip install ./submodules/diff-gaussian-rasterization

pip install ./submodules/simple-knn
```
## Compile CUDA Extensions 
```bash
pip install ./submodules/simple-knn

pip install ./bvh

pip install ./r3dg-rasterization

pip install ./diff-gaussian-rasterization
```

## Fix Common Runtime Error (Optional but Recommended)
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Verify Installation
```bash
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```
Torch: 2.1.2+cu118 \
CUDA available: True

## Run in Narrow-Mode
### Train with terminal
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --eval -s "datasets/real_image/H_v2/H-RF-RCS" -m output/real_image/H_v2/render_RF -c output/real_image/H_v2/3dgs/chkpnt30000.pth --save_training_vis --freq wideband --position_lr_init 0 --position_lr_final 0 --normal_lr 0 --sh_lr 0.0025 --opacity_lr 0.005 --scaling_lr 0 --rotation_lr 0 --iterations 40000 -t render_RF --save_training_vis_iteration 400 --lambda_env_smooth 0.01 
```


### Evalualuate
```bash
CUDA_VISIBLE_DEVICES=0 python eval_nvs_RF_real.py --eval -m output/real_image/H_v2/render_RF -c output/real_image/H_v2/render_RF/chkpnt40000.pth -t render_RF  
```


## Run in Wideband Mode
```bash
CUDA_VISIBLE_DEVICES=0 python wideband_train.py --eval -m output/real_image/H_v2/render_RF_bb_v48 -s datasets/real_image/H_v2/H-RF-RCS --RF_iterations 5000 --position_lr_init 0 --position_lr_final 0 --normal_lr 0 --sh_lr 0.0 --opacity_lr 0.0 --scaling_lr 0 --rotation_lr 0 --power_scale 0.01
```
```bash
CUDA_VISIBLE_DEVICES=0 python wideband_test.py --eval -m output/real_image/H_v2/render_RF_bb_v48 -c output/real_image/H_v2/render_RF_v7/chkpnt32000.pth -s datasets/real_image/H_v2/H-RF-RCS -t1 render_RF -t2 render_RF_bb --power_scale 0.01 
```




























