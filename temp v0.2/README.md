We aim to improve results from the article "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa" (Yeh et al, 2020) published in *Nature Communications* on May 22, 2020 ([link](https://www.nature.com/articles/s41467-020-16185-w)). 

## Installation
conda env create -f env.yml

AlLso, to download cv2
pip install opencv-python

Furthermore, on linux/windows
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


## Preparing data
1. Copy the following survey data from their github [link](https://github.com/sustainlab-group/africa_poverty/tree/master/data) to our folder data/raw: dhs_wealth_index.csv, lsms_wealth_index.csv.

2. Rearrange data by running dataOvPrep.ipynb.
3. Download all data by running all dataDownload_*.ipynb
4. Clean up data by running dataCleanup.ipynb


## Model training
Using entropy of image in the input.
```
python main.py --n_epoch 4
```
Doesn't seem to work well so far.