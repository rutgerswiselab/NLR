# Neural Logic Reasoning

This repository includes our implementation for Neural Logic Reasoning (NLR):

*Shaoyun Shi, Hanxiong Chen, Weizhi Ma, Jiaxin Mao, Min Zhang, Yongfeng Zhang. 2020. [Neural Logic Reasoning](https://arxiv.org/pdf/2008.09514.pdf). 
In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20).*

## Refernece

For inquiries contact Shaoyun Shi (shisy13@gmail.com) or Yongfeng Zhang (yongfeng.zhang@rutgers.edu)

```
@inproceedings{shi2020neural,
  title={Neural Logic Reasoning},
  author={Shi, Shaoyun and Chen, Hanxiong and Ma, Weizhi and Mao, Jiaxin and Zhang, Min and Zhang, Yongfeng},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={1365--1374},
  year={2020}
}
```

## Environments

Python 3.7.3

Packages: See in [requirements.txt](https://github.com/rutgerswiselab/NLR/blob/master/requirements.txt)

```
numpy==1.18.1
torch==1.0.1
pandas==0.24.2
scipy==1.3.0
tqdm==4.32.1
scikit_learn==0.23.1
```



## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/rutgerswiselab/NLR/tree/master/dataset)

- **Logic-1k**: Dataset for solving logical equations with 1k variables

- **Logic-10k**: Dataset for solving logical equations with 10k variables

- **ML-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/100k/). 

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 

- The codes for processing the data can be found in [`./src/datasets/`](https://github.com/rutgerswiselab/NLR/tree/master/src/datasets)

    

## Example to run the codes

-   Some running commands can be found in [`./command/command.py`](https://github.com/rutgerswiselab/NLR/blob/master/command/command.py)
-   For example:

```
# Neural Logic Reasong for recommendation on ML-100k dataset
> cd NLR/src/
> python main.py --rank 1 --model_name NLRRec --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --max_his 10 --sparse_his 0 --neg_his 1 --l2 1e-4 --r_logic 1e-06 --r_length 1e-4 --random_seed 2018 --gpu 0
```

