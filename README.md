# Genie 2: Designing and Scaffing Proteins at the Scale of the Structural Universe

This repository provides the implementation code for our [preprint](https://arxiv.org/abs/2405.15489), including training and inference code, as well as model weights. For the in-silico evaluation pipeline, which is used to assess the designability, diversity and novelty of our generated structures, we provide them in a seperate [repository](https://github.com/aqlaboratory/insilico_design_pipeline) since it is independent of Genie 2 and could be applicable for evaluating other protein structure diffusion models. Below provides an illustration of the Genie 2 sampling process for both unconditional generation and motif scaffolding.

![](https://github.com/aqlaboratory/genie2/blob/main/sampling.gif)

## Setup

Set up Genie 2 by running
```
git clone https://github.com/aqlaboratory/genie2.git
cd genie2
pip install -e .
```
This would clone the repository, install dependencies and set up Genie 2 as a packge.

## Training

### Dataset

In `data/afdbreps_l-256_plddt_80`, we provide an index file named `index.txt`. It contains a list of 588,571 AFDB filenames, each of which correponds to a structure in the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), specifically in the format `https://alphafold.ebi.ac.uk/files/[FILENAME].pdb`. These structures are representative structures from the [FoldSeek-clustered AFDB](https://cluster.foldseek.com/) dataset (filtered with a maximum sequence length of 256 and a minimum pLDDT of 80) and are used for the training of Genie 2. To set up for this dataset, create a subdirectory named `pdbs` under `data/afdbreps_l-256_plddt_80` and download this set of structures from AFDB into this `pdbs` subdirectory. Our default training data directory is set to `data/afdbreps_l-256_plddt_80/pdbs` and this could be changed by using the `dataDirectory` key in the configuration file.

### Training

To train a model, create a directory `runs/[RUN_NAME]` and create a configuration file with name `configuration` under this directory. An example is provided in `runs/example` and a complete list of configurable parameters could be found in `genie/config.py`. Note that in the configuration file, name should match with RUN_NAME in order to log into the correct directory. To start training, run
```
python genie/train.py --devices [NUM_DEVICES] --num_nodes [NUM_NODES] --config runs/[RUN_NAME]/configuration
```

## Sampling

### Directory Setup

To sample using a model, create a directory `results/[MODEL_NAME]`, which consists of
-	the model configuration file named `configuration`
-	a subdirectory named `checkpoints`, where each file is named `epoch=[EPOCH].ckpt` and contains the model weight checkpointed at the specified epoch.

An example is provided under `results/base`, which contains the configuration and checkpoints for our trained model. The checkpoints are uploaded via Git LFS (Large File Storage). If LFS is not installed in your machine, [install LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) and run the command below at the main directory.  

```
git lfs install
git pull lfs
```

For results reported in our manuscript, we used the 40-epoch checkpoint for unconditional generation and 30-epoch checkpoint for motif scaffolding.

### Unconditional Generation

Perform unconditional sampling by running

```
python genie/sample_unconditional.py --name [NAME] --epoch [EPOCH] --scale [SCALE] --outdir [OUTDIR]
``` 

The list of parameters are summarized in the following table.

| Parameter | Type   | Description | Required | Default |
| :-------: | :----: | ---------- | :------: | :-----: |
| `name` | string | Model name | :heavy_check_mark: | |
| `epoch` | int | Model epoch | :heavy_check_mark: | |
| `scale` | float | Sampling noise scale (between 0 and 1, inclusive) | :heavy_check_mark: | |
| `outdir` | string | Output directory |:heavy_check_mark: | |
| `rootdir` | string | Model root directory | | results |
| `num_samples` | int | Number of samples per length |  | 5 |
| `batch_size` | int | Batch size |  | 4 |
| `min_length` | int | Minimum sequence length |  | 50 |
| `max_length` | int | Maximum sequence length |  | 256 |
| `length_step` | int | Sequence length step | | 1 |
| `num_devices` | int | Number of GPUs | | 1 |

To balance the sampling time across GPUs, we shuffle the generation tasks. To avoid this behavior and sample in increasing order of length, add the flag `--sequential_order`.

To reproduce our unconditional generation, run the command below. 

```
python genie/sample_unconditional.py --name base --epoch 40 --scale 0.6 --outdir results/base/outputs
```

### Motif Scaffolding

We provide the benchmark datasets for single-motif and multi-motif scaffolding under `data/design25` and `data/multimotifs`, respectively. The format of a motif scaffolding problem definition file is described [here](https://github.com/aqlaboratory/genie2?tab=readme-ov-file#format-of-motif-scaffolding-problem-definition-file).

Perform motif scaffolding by running

```
python genie/sample_scaffold.py --name [NAME] --epoch [EPOCH] --scale [SCALE] --outdir [OUTDIR]
```

The list of parameters are summarized in the following table.

| Parameter | Type   | Description | Required | Default |
| :-------: | :----: | ---------- | :------: | :-----: |
| `name` | string | Model name | :heavy_check_mark: | |
| `epoch` | int | Model epoch | :heavy_check_mark: | |
| `scale` | float | Sampling noise scale (between 0 and 1, inclusive) | :heavy_check_mark: | |
| `outdir` | string | Output directory | :heavy_check_mark: | |
| `rootdir` | string | Model root directory | | results |
| `num_samples` | int | Number of samples per length | | 100 |
| `batch_size` | int | Batch size | | 4 |
| `motif_name` | string | Name of motif-scaffolding problem (Automatically <br>evaluate all problems in `datadir` if unspecified) | | |
| `datadir` | string | Directory of motif-scaffolding problems | | data/design25 |
| `num_devices` | int | Number of GPUs | | 1 |

To reproduce our single-motif scaffolding, run

```
python genie/sample_scaffold.py --name base --epoch 30 --scale 0.4 --outdir results/base/design25 --num_samples 1000
``` 

To reproduce our multi-motif scaffolding, run

```
python genie/sample_scaffold.py --name base --epoch 30 --scale 0.4 --outdir results/base/multimotifs --datadir data/multimotifs --num_samples 1000
```

## Format of Motif Scaffolding Problem Definition File

A motif scaffolding problem definition file consists of two parts: the first part defines motif scaffolding specifications, including arrangements of motif segments, minimum/maximum length of each scaffold segment and minimum/maximum sequence length; the second part defines motif structure in a PDB format. The table below details the format for defining motif scaffolding specifications.

| Specification | Column | Data | Justification | Data Type |
| :-------: | ------ | ---------- | :------: | :-----: |
| Motif segment | 1-16 | "REMARK 999 INPUT" | | string <tr></tr>|
|               | 19 | Chain index of motif segment in the PDB file | | string <tr></tr>|
|               | 20-23 | Starting residue index of motif segment in the PDB file | right | int <tr></tr>|
|               | 24-27 | Ending residue index of motif segment in the PDB file | right | int <tr></tr>|
|               | 29 | Motif group that the segment belongs to (default to 'A' if unspecified) | | string |
| Scaffold segment | 1-16 | "REMARK 999 INPUT" | | string <tr></tr>|
|                  | 20-23 | Minimum length of scaffold segment | right | int <tr></tr>|
|                  | 24-27 | Maximum length of scaffold segment | right | int |
| Minimum <br>sequence length | 1-31 | "REMARK 999 MINIMUM TOTAL LENGTH" | | string <tr></tr>|
|                             | 38-40 | Minimum sequence length | left | int |
| Maximum <br>sequence length | 1-31 | "REMARK 999 MAXIMUM TOTAL LENGTH" | | string <tr></tr>|
|                             | 38-40 | Maximum sequence length | left | int |

One example of motif scaffolding specifications is provided below.

```
REMARK 999 NAME   1PRW_two
REMARK 999 PDB    1PRW
REMARK 999 INPUT      5  20
REMARK 999 INPUT  A  16  35 A
REMARK 999 INPUT     10  25
REMARK 999 INPUT  A  52  71 A
REMARK 999 INPUT     10  30
REMARK 999 INPUT  A  89 108 B
REMARK 999 INPUT     10  25
REMARK 999 INPUT  A 125 144 B
REMARK 999 INPUT      5  20
REMARK 999 MINIMUM TOTAL LENGTH      120
REMARK 999 MAXIMUM TOTAL LENGTH      200
```

There are two limitations with our current format:
- When defining motif structures in a PDB format, the residue order should match with the order in motif scaffolding specification.
- PDB name is not specified when defining motif segment in motif scaffolding specification.

We intend to optimize our code to address these two limitations in the future.
