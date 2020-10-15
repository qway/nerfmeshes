# NeRF: Neural Radiance Fields Extension

Extension of the NeRF (Neural Radiance Fields) method using PyTorch (PyTorch Lightning).
![title](misc/pipeline.png)

Based on the official implementation: [nerf](https://github.com/bmild/nerf)

### [Paper](https://drive.google.com/drive/folders/1nssFxbSrTGOkNVSN3klsHh9RJg_ik5YF?usp=sharing) | [Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

 [Benedikt Wiberg](https://github.com/qway) <sup>1</sup>,
 [Cristian Chivriga](https://github.com/DomainFlag) <sup>1</sup>,
 [Marian Loser](https://github.com/Discusxl) <sup>1</sup>,
 [Yujiao Shentu](https://github.com/styj5) <sup>1</sup><br> <sup>1</sup>TUM

## NeRF :fireworks:

NeRF (Neural radiance field) optimizes directly the parameters of continuous 5D scene representation by minimizing the error of view synthesis with the captured images.

## Extension of NeRF :sparkler:

The project is an extension and improvement upon the original method NeRF for neural rendering view-synthesis designed for rapid prototyping and experimentation. Main improvements are: 
 - Scene encoding through unstructured radiance volumes.
 - Mesh reconstruction with appearance through informed re-sampling based on the inverse normals of the scene geometry. Additionally, we present both sparse and dense reconstruction through Poisson Surface Reconstruction and Marching Cubes. 
 - Modular implementation which is 1.4x faster and twice as much memory efficient then the base implementation [NeRF-PyTorch](https://github.com/krrish94/nerf-pytorch).

## Get started

Install the dependencies via: 

### Option 1: Using pip

In a new `conda` or `virtualenv` environment, run

```bash
pip install -r requirements.txt
```

### Option 2: Using poetry

In the root folder, run

```bash
poetry install
source .venv/bin/activate
```

### Option 3: Get started on Google Colab

In the root folder, run the script considering the dataset folder is named data/ inside your Drive folder (otherwise change it accordingly):

```bash
. ./script.sh
```

## Data

#### Synthetic dataset and LLFF

Checkout the provided data links [NeRF Original](https://github.com/bmild/nerf).

#### Your own data with [colmap](https://colmap.github.io/)
<details> <summary>Steps</summary>
   
1. Install [COLMAP](https://github.com/colmap/colmap) following [installation guide](https://colmap.github.io/install.html)
2. Prepare your images in a folder (60-80). Make sure that auto-focus is turned off.
3. Run `python data/colmap_convert.py your_images_dir_path`
4. Edit the config file e.g. `config/colmap.yml` with generated dataset path.
</details>

## Running code

#### Run training

Get to know the configuration files under the `src/config` and get started running your own experiments by creating new ones.

The training script can be invoked by running
```bash
python train_nerf.py --config config/lego.yml
```

To resume training from latest checkpoint:
```bash
python train_nerf.py --config config/lego.yml --load-checkpoint your_log_dir_path
```

#### Extracting Mesh with appearance

Get to know the configuration files under the `src/config` and get started running your own experiments by creating new ones.

The training script can be invoked by running
```bash
python mesh_nerf.py --checkpoint your_log_checkpoint_dir_path --config config/lego-high-res.yml --save-dir ../data/meshes --limit 1.2 --res 256 --iso-level 16
```


#### See your results :star:

If `tensorboard` is properly installed, check real-time your results on `localhost:6006`:

```bash
tensorboard --logdir logs/... --port 6006
``` 

## Credits

Based on the existent work:

- [NeRF Original](https://github.com/bmild/nerf) - Original work in TensorFlow, please refer also the paper for extra information.
- [NeRF PyTorch](https://github.com/krrish94/nerf-pytorch) - PyTorch base code for the current repo.
