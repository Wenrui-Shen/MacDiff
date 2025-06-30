# [ECCV 2024] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion

<!-- 🤗 -->
<font size=4><div align='center'>[[📝 Paper](https://arxiv.org/abs/2409.10473)] [[📄 Supplementary](https://lehongwu.github.io/ECCV24MacDiff/macdiff-supp.pdf)] [[🚀 Website](https://lehongwu.github.io/ECCV24MacDiff/index.html)]</div></font>


<div align="center">
<div>
  <font size=5>
    <p>🎉  <b>Accepted by ECCV 2024</b></p>
  </font>
</div>
</div>

<div align="center">
  <img src="assets/overview_macdiff.png" alt="Method Overview" width="100%">
</div>


## Installation

```bash
conda create -n macdiff python=3.8

conda activate macdiff

pip install -r requirements.txt
```


## Data Preparation
Please follow the data preparation process of [MAMP](https://github.com/maoyunyao/MAMP).

## Training and Testing
Please refer to the bash scripts. Before running the scripts, you may:

1. Check the default configurations in the yaml files. You can also overwrite them in the bash scripts.

2. Change the paths for saving your checkpoints and logs.

If you find any problems with the code, please feel free to open an issue or contact us by sending an email to aladonwlh[AT]stu.pku.edu.cn.

## Citation
If you find this work useful for your research, please consider citing our work:
```
@inproceedings{wu2024macdiff,
  title={MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion},
  author={Wu, Lehong and Lin, Lilang and Zhang, Jiahang and Ma, Yiyang and Liu, Jiaying},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgment
The framework of our code is based on [MAE](https://github.com/facebookresearch/mae), [MAMP](https://github.com/maoyunyao/MAMP).