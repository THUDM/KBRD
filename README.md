# KBRD

### [arXiv](https://arxiv.org/abs/1908.05391)

Towards **K**nowledge-**B**ased **R**ecommender **D**ialog System.<br>
[Qibin Chen](https://www.qibin.ink), [Junyang Lin](https://justinlin610.github.io), Yichang Zhang, Ming Ding, [Yukuo Cen](https://sites.google.com/view/yukuocen), [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/).<br>
In EMNLP-IJCNLP 2019

## Prerequisites

- Linux
- Python 3.6
- PyTorch 1.2.0

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/KBRD
cd KBRD
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

- We use the **ReDial** dataset, which will be automatically downloaded by the script.
- Download the refined knowledge base (dbpedia) used in this paper [[Google Drive]](https://drive.google.com/open?id=1WqRoQAxH_kdoJpbYVsFF0EN4ZJxiiDB2) [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/6af126bdccc44352bfee/?dl=1). Decompress it and get the `dbpedia/` folder, which should contain two files `mappingbased_objects_en.ttl` and `short_abstracts_en.ttl`.
- Download the proprocessed extracted entities set [[Google Drive]](https://drive.google.com/open?id=1OG-kNIeUi3i0UDNhJVMEnia9JeRAHVXB) [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/88ac4b7eab6c416ca74f/?dl=1) and put it under `<path/to/KBRD/data/redial/`.

### Training

To train the recommender part, run:

```bash
bash scripts/both.sh <num_exps> <gpu_id>
bash scripts/baseline.sh <num_exps> <gpu_id>
```

To train the dialog part, run:

```bash
bash scripts/t2t_rec_rgcn.sh <num_exps> <gpu_id>
```

### Logging

TensorBoard logs and models will be saved in `saved/` folder.

### Evaluation

- `show_bias.py` is used to show the vocabulary bias of a specific movie (like in Table 4)

TODO

If you have difficulties to get things working in the above steps, please let us know.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{chen2019towards,
  title={Towards Knowledge-Based Recommender Dialog System},
  author={Chen, Qibin and Lin, Junyang and Zhang, Yichang and Ding, Ming and Cen, Yukuo and Yang, Hongxia and Tang, Jie},
  journal={arXiv preprint arXiv:1908.05391},
  year={2019}
}
```
