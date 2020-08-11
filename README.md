# KBRD

### [arXiv](https://arxiv.org/abs/1908.05391)

Towards **K**nowledge-**B**ased **R**ecommender **D**ialog System.<br>
[Qibin Chen](https://www.qibin.ink), [Junyang Lin](https://justinlin610.github.io), Yichang Zhang, Ming Ding, [Yukuo Cen](https://sites.google.com/view/yukuocen), [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/).<br>
In EMNLP-IJCNLP 2019

* **New: The current SOTA on the ReDial dataset is https://arxiv.org/pdf/2007.04032.pdf. Don't miss out that great work!**
* New: code and README are improved.
* We curated a paper list for NLP + Recommender System at https://github.com/THUDM/NLP4Rec-Papers. Contributions are welcome.

## Prerequisites

- Linux
- Python 3.6
- PyTorch >= 1.2.0

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
- Download the refined knowledge base (dbpedia) used in this paper [[Google Drive]](https://drive.google.com/open?id=1WqRoQAxH_kdoJpbYVsFF0EN4ZJxiiDB2) [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/6af126bdccc44352bfee/?dl=1). Decompress it and get the `<path/to/KBRD/dbpedia/>` folder, which should contain two files `mappingbased_objects_en.ttl` and `short_abstracts_en.ttl`.
- Download the proprocessed extracted entities set [[Google Drive]](https://drive.google.com/open?id=1OG-kNIeUi3i0UDNhJVMEnia9JeRAHVXB) [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/88ac4b7eab6c416ca74f/?dl=1) and put it under `<path/to/KBRD/data/redial/`.

### Training

1. To train the recommender part, run:

```bash
bash scripts/both.sh <num_exps> <gpu_id>
(optional) bash scripts/baseline.sh <num_exps> <gpu_id>
```

2. To train the dialog part, run:

```bash
bash scripts/t2t_rec_rgcn.sh <num_exps> <gpu_id>
```

The test results are displayed at the end of training and can also be found at `saved/<model_name>.test`.

### Logging

Training outputs, TensorBoard logs and models files are be saved in `saved/` folder.

### Evaluation

1. `scripts/score.py` is used to hypothesis testing the significance of improvement between different models. To use, first run multiple experiments with `num_exps > 1`, for example:

```
bash scripts/both.sh 2 <gpu_id>
bash scripts/baseline.sh 2 <gpu_id>
```

Then,

```bash
python scripts/score.py --name-1 saved/release_baseline --name-2 saved/both_rgcn --num 2 --metric recall@50
```
where you should remove the trailing `_0`, `_1` automatically added to the model names, `nums` should be set the same as `num_exps` above, and `recall@50` can be replaced with other evaluation metrics in the paper.

Sample output:

```
[0.298, 0.2918]
0.2949
0.0031
[0.3417, 0.3369]
0.3393
0.0024
Ttest_indResult(statistic=-11.325204070341204, pvalue=0.007706635327863829)
```

2. `scripts/display_model.py` is used to generate responses.

```bash
python scripts/display_model.py -t redial -mf saved/transformer_rec_both_rgcn_0 -dt test
```

Example output (\[TorchAgent\] is our model output):

```
~~
[eval_labels_choice]: Oh, you like scary movies?
I recently watched __unk__
[movies]:
  37993
[redial]: 
Hello!
Hello!
What kind of movies do you like?
I am looking for a movie recommendation.   When I was younger I really enjoyed the __unk__
[label_candidates: 3|37993|50395||Oh, you like scary movies?
I recently watched __unk__]
[eval_labels: Oh, you like scary movies?
I recently watched __unk__]
   [TorchAgent]: have you seen "The Shining  (1980)" ?
~~
```

3. `scripts/show_bias.py` is used to show the vocabulary bias of a specific movie (like the qualitative analysis in Table 4)

```bash
python scripts/show_bias.py -mf saved/transformer_rec_both_rgcn_0
```

## ❗ Common Q&A

1. Understanding model outputs.
Please see https://github.com/THUDM/KBRD/issues/15#issuecomment-636367468.

2. Adapting this code to other datasets.
It is not straightforward for this code to be run on other datasets currently.
The main reason is that we cached the entity linking process in KBRD for ReDial. Please see https://github.com/THUDM/KBRD/issues/10#issuecomment-585261932 for details.

3. Why the recommender and the dialog part are trained separatedly?
Please refer to https://github.com/THUDM/KBRD/issues/9#issuecomment-556988541 for detailed explanation.

If you have additional questions, please let us know.

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
