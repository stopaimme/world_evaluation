# World Model Evaluation


## Installation

1. Clone the repo

```sh
git clone https://github.com/stopaimme/world_evaluation.git
cd world_evaluation
```

2. Create the environment

```sh
conda create -n world_evaluation python=3.10
conda activate world_evaluation

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
2. setup droid-slam and vggt

```sh
mkdir thirdparty
cd thirdparty
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
git clone https://github.com/facebookresearch/vggt.git
cd DROID-SLAM
pip install thirdparty/lietorch --no-build-isolation
pip install thirdparty/pytorch_scatter --no-build-isolation
pip install -e . --no-build-isolation
cd ../..
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
```

## Running

```sh
python evaluation.py --reference_dir GT_case1 --generated_dir generated_case1

GT_case1
├── images
│   ├── frame1.png                   
│   ├── frame2.png
├── transform.json

generated_case1
├── frame1.png
├── frame2.png
```
