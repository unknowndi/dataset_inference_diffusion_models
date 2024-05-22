# CDI: Copyrighted Data Identification in Diffusion Models

### Keywords
 
dataset inference, diffusion models, copyright, intellectual property, membership inference

### TL;DR

We demonstrate that existing membership inference attacks are not effective in confidently identifying (illicit) use of data to train diffusion models and instead propose the first dataset inference-based method to achieve this goal.

## Abstract

Diffusion Models (DMs) benefit from large and diverse datasets for their training. Since this data is often scraped from the internet without permission from the data owners, this raises concerns about copyright and intellectual property protections. While (illicit) use of data is easily detected for training samples perfectly re-created by a DM at inference time, it is much harder for data owners to verify if their data was used for training when the outputs from the suspect DM are not close replicas. Conceptually, membership inference attacks (MIAs), which detect if a given data point was used during training, present themselves as a suitable tool to address this challenge. However, we show that existing MIAs are not practical due to their low confidence on individual data points’ membership status. To overcome this limitation, we propose Copyrighted Data Identification (CDI), a framework for data owners to confidently verify whether their data was used to train a given DM. CDI relies on dataset inference techniques, i.e., instead of using the membership signal from a single data point, CDI leverages the fact that most data owners, such as website providers, bloggers, artists, or publishers own datasets with multiple publicly exposed data points which might all be included in the training of a given DM. By using the signal from existing MIAs and new handcrafted methods to extract features from these datasets, feeding them to a scoring model, and applying rigorous statistical testing, CDI allows data owners with as little as 30 data points to identify with a confidence of more than 99% whether their data was used to train a DM. Thereby, CDI represents a valuable tool for data owners to claim illegitimate use of their copyrighted data. 

## Codebase setup

### Environment configuration

A suitable conda environment named `cdi` can be created and activated with:

```
conda env create -f environment.yaml
conda activate cdi
```

#### Troubleshooting

In case of GBLICXX import error run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[YOUR_PATH_TO_CONDA]/envs/cdi/lib` (based on [this](https://stackoverflow.com/a/71167158))

### Downloading models

```
gdown https://drive.google.com/drive/folders/143b1wF1iWEU2DASTk-sfTRwW7KiEC3IX?usp=sharing --folder
```

### Downloading data and data preparation

* ImageNet: Download [train](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) and [validation](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) ImageNet LSVRC 2012 splits.

* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract annotations features according to `helper_scripts/uvit_extract_embeddings.py`. 

## Experiments overview

