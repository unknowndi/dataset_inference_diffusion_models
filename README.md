# Dataset Inference for Diffusion Models

### Requirements
A suitable conda environment named test_ldm can be created and activated with:

```
conda env create -f environment.yaml
conda activate test_ldm
```

In case of GBLICXX import error run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[YOUR_PATH_TO_CONDA]/envs/test_ldm/lib` (based on [this](https://stackoverflow.com/a/71167158))

### Downloading models

```
gdown https://drive.google.com/drive/folders/143b1wF1iWEU2DASTk-sfTRwW7KiEC3IX?usp=sharing --folder
```

### Downloading data and data preparation
* ImageNet: Download [train](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) and [validation](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) ImageNet LSVRC 2012 splits.

* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract annotations features according to `helper_scripts/uvit_extract_embeddings.py`. 