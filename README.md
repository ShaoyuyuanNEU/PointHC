# Exploring High-contrast Areas Context for 3D Point cloud Segmentation via MLP-driven Discrepancy Mechanism

## Installation
We provide a simple bash file to install the environment:

The pre-trained models for this project can be found on [Baidu Netdisk](https://pan.baidu.com/s/1_0zXnFiIEEEDuHooG7ZySw password: 5jbs)

```
cd PointMLPD
source update.sh
source install.sh
```
Cuda-11.3 is required.

scikit-learn==1.0.2
pickleshare==0.7.5
ninja==1.10.2.3
gdown
easydict==1.9
PyYAML==6.0
protobuf==3.19.4
tensorboard==2.8.0
termcolor==1.1.0
tqdm==4.62.3
multimethod==1.7
h5py==3.6.0
matplotlib==3.5.1
wandb
pyvista
setuptools==59.5.0
Cython==0.29.28
pandas
deepspeed
shortuuid

# for docs
mkdocs-material
mkdocs-awesome-pages-plugin
mdx_truly_sane_lists


#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc


# The following 4 lines are only for slurm machines. uncomment if needed.  
# export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
# module purge
# module load cuda/11.3.1
# module load gcc/7.5.0

# download openpoints
# git submodule add git@github.com:guochengqian/openpoints.git
git submodule update --init --recursive
git submodule update --remote --merge # update to the latest version

# install PyTorch
conda deactivate
conda env remove --name openpoints
conda create -n openpoints -y python=3.7 numpy=1.20 numba
conda activate openpoints

# please always double check installation for pytorch and torch-scatter from the official documentation
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

pip install -r requirements.txt

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../

## Acknowledgment

Our code refers to the work [PointNext](https://github.com/guochengqian/PointNeXt)

