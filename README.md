
# Step 1
```
git clone --recurse-submodules git@github.com:huangjund/HandCeption.git
```

# Step 2
system: ubuntu 24.04.1
GPU: 2080Ti
Cuda: 12.4
torch version: 2.6.1
```angular2html
conda create -n handcept python=3.10 -y
pip3 install torch torchvision torchaudio
git clone https://github.com/ethnhe/FFB6D.git
cd FFB6D
pip3 install -r requirement.txt

# install apex:
git clone https://github.com/NVIDIA/apex
cd apex
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

#install normalSpeed:
git clone https://github.com/hfutcgncas/normalSpeed.git
cd normalSpeed/normalSpeed
python3 setup.py install --user
cd ..

sudo apt install python3-tk


```

## check opencv in python
```angular2html
import cv2
print(cv2.__file__)
print(cv2.__version__)
```
If OpenCV is installed via pip, it will return a path like:
```
/home/jd/miniconda3/envs/handcept/lib/python3.10/site-packages/cv2/__init__.py
```
## check opencv system installation
```angular2html
/home/jd/miniconda3/envs/handcept/lib/python3.10/site-packages/cv2/__init__.py
```
## Install opencv (if not installed)
```angular2html
sudo apt update
sudo apt install libopencv-dev
# verify installation
pkg-config --modversion opencv4
```
# Train the model

```angular2html
n_gpu=8
cls='ape'
torchrun --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls
```

# TODO

1. Substitute apex from nvidia github repo to pytorch package.
2. 