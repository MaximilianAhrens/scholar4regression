# Scholar4Regression

This is an extension of the [Scholar](https://arxiv.org/abs/1705.09296) model, which is a tool for modeling documents with metadata.
The original model features a classifier network for downstream tasks. This extension adds a regression network to it, in order to allow applications for a wider range of supervised downstream tasks.

The implementation is done based on the tensorflow version of the original model.

## Requirements:

- python3
- tensorflow 1.15.0
- numpy
- scipy
- pandas
- gensim

It is recommended that you install the requirements using Anaconda. Specifically, you can use the following commands to create a new environment, activate it, and install the necessary packages:

`conda create -n scholar python=3`

`source activate scholar`

`conda install tensorflow==1.15.0`

`conda install numpy scipy pandas gensim`

Once the necessary packages are installed, there is no need to compile or install this repo.


## Usage & Tutorial:
The github page of the [original model](https://github.com/dallascard/scholar) provides a great description of the model's functionalities. It also provides you with a tutorial which is worth exploring. Read those instructions first to familiarse yourself with the model's user interface.

## Unsupervised topic learning:
Example for running SCHOLAR with k = 10 topics in an unsupervised fashion (based on original model)
`python run_scholar_tf.py input_dir -k 10 --test test --topic-covars`

## Supervised topic learning + classification:
Example for running SCHOLAR with k = 10 topics in a supervised fashion with a classification task (based on original model).
Adding covariates (--topic-covars) is optional.

`python run_scholar_tf.py input_dir -k 10 --test test --topic-covars covar_file --labels target_variable_file --task "class"`

## Supervised topic learning + regression:
Example for running SCHOLAR with k = 10 topics in a supervised fashion with a regression task (based on model extension).
Adding covariates (--topic-covars) is optional.

Regression task with linear regression network: $y = W\theta$
`python run_scholar_tf.py input_dir -k 10 --test test --topic-covars covar_file --labels target_variable_file --task "reg" --r-layers 0`
Regression task with non-linear regression network with 2 hidden layers: $y = f(W,\theta)$
`python run_scholar_tf.py input_dir -k 10 --test test --topic-covars covar_file --labels target_variable_file --task "reg" --r-layers 2`

Options for the regression network are:
--r-layers [0|1|2], where 0 is the default and corresponds to a linear regression.



## References

If you find this repo useful, please be sure to cite the following publication:

* Dallas Card, Chenhao Tan, and Noah A. Smith. Neural Models for Documents with Metadata. In *Proceedings of ACL* (2018). [[paper](https://www.cs.cmu.edu/~dcard/resources/ACL_2018_paper.pdf)] [[supplementary](https://www.cs.cmu.edu/~dcard/resources/ACL_2018_supplementary.pdf)] [[BibTeX](https://github.com/dallascard/scholar/blob/master/scholar.bib)]

