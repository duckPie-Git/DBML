# DBML
This is the python implementation of DBML model for paper "Dynamic Bayesian Metric Learning for Personalized Product Search
(Teng Xiao<sup>\*</sup>, Jiaxin Ren<sup>\*</sup>, Shangsong Liang and Zaiqiao Meng)" 


# Introduction
DBML is a novel probabilistic metric learning approach that is able to avoid the contradicts, keep
the triangle inequality in the latent space, and correctly utilize implicit feedbacks. The inferred dynamic semantic representations of entities collaboratively inferred in a unified form by our DBML can benefit not only for
improving personalized product search, but also for capturing the affinities between users, products and words. Please refer to the paper for further details.

# Requirements
* pytorch(0.4 or later)
* nltk
* tqdm
* dateutil
* gzip





# Datasets

Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (In our paper, we used 5-core review data and metedata).


# Run
Run train/train_baseline.py for static model.
```shell
python train_baseline.py --cuda --data_name dataname
```
Run train/train.py for dynamic model.
```shell
python train.py --cuda --data_name dataname
```
Where the data_name can be 'Electronics', 'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry' or 'Toys and Games'.



# Citation
if you want to use our codes in your research, please cite:
```
@inproceedings{dbml/cikm/2019,
  title={Dynamic Bayesian Metric Learning for Personalized Product Search},
  author={Xiao, Teng and Ren, Jiaxin and Liang, Shangsong and Meng, Zaiqiao},
  booktitle={Proceedings of the  28th ACM International Conference on Information and Knowledge Management},
  year={2019},
  organization={ACM}
}
```
