# SEER: Transformer-based Robust Time Series Forecasting via Automated Patch Enhancement and Replacement

 [![ICML](https://img.shields.io/badge/ICML'26-SEER-orange)](https://arxiv.org/pdf/2602.00589)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)

This code is the official PyTorch implementation of our ICML'26 paper **SEER: Transformer-based Robust Time Series Forecasting via Automated Patch Enhancement and Replacement**.

<div align="center">
<img alt="Logo" src="docs/figures/overview.png" width="100%"/>
</div>


## Quickstart

> [!IMPORTANT]
> this project is fully tested under python 3.8, it is recommended that you set the Python version to 3.8.
1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model

- To see the model structure of SEER,  [click here](./ts_benchmark/baselines/seer/models/seer_model.py).
- We provide all the experiment scripts for SEER and other baselines under the folder `./scripts/SEER_script`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/SEER_script/ETTh1/SEER.sh
```

## Results
<div align="center">
<img alt="Logo" src="docs/figures/main-results.png" width="100%"/>
</div>


## Citation

If you find this repo useful, please cite our paper.

```

@inproceedings{qiu2026seer,
  title        =   {SEER: Transformer-based Robust Time Series Forecasting via Automated Patch Enhancement and Replacement},
  author       =   {Qiu, Xiangfei and Liu, Xvyuan and Shen, Tianen and Wu, Xingjian and Cheng, Hanyin and Yang, Bin and Hu, Jilin},
  booktitle    =   {ICML},
  year         =   {2026}
}
```



## Contact

If you have any questions or suggestions, feel free to contact:

- [Xiangfei Qiu](https://qiu69.github.io/) (xfqiu@stu.ecnu.edu.cn)


Or describe it in Issues.
