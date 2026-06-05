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

Torch: 2.1.2+cu118
CUDA available: True

## Run





























