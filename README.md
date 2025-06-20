# Grapose

Gyoung Jin Park, Gyounyoung Heo, Dasom Noh, Yeongyeong Son, Sunyoung Kwon

## Installation

```sh
conda create -n grapose python=3.12 -y
conda activate grapose

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html
pip install rdkit deepchem tqdm 

# consider pH environments, logger
pip install dimorphite-dl knockknock
conda install -c conda-forge pymol-open-source
```

---
## Datasets  <a name="datasets"></a>

The files in `data` contain the splits used for the various datasets. Below instructions for how to download each of the different datasets used for training and evaluation:

 - **PDBBind:** download the processed complexes from [zenodo](https://zenodo.org/record/6408497), unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`.
 - **BindingMOAD:** download the processed complexes from [zenodo](https://zenodo.org/records/10656052) under `BindingMOAD_2020_processed.tar`, unzip the directory and place it into `data` such that you have the path `data/BindingMOAD_2020_processed`.
 - **DockGen:** to evaluate the performance of `DiffDock-L` with this repository you should use directly the data from BindingMOAD above. For other purposes you can download exclusively the complexes of the DockGen benchmark already processed (e.g. chain cutoff) from [zenodo](https://zenodo.org/records/10656052) downloading the `DockGen.tar` file.
 - **PoseBusters:** download the processed complexes from [zenodo](https://zenodo.org/records/8278563).
 - **van der Mers:** the protein structures used for the van der Mers data augmentation strategy were downloaded [here](https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz).



## Contact (Questions/Bugs/Requests)
Please submit a GitHub issue or contact me [rudwls2717@pusan.ac.kr](rudwls2717@pusan.ac.kr)

## Acknowledgements
Thank you for our [Laboratory](https://www.k-medai.com/).

If you find this code useful, please consider citing my work.