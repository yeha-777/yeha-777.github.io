---
layout: post
title: CausalML installation
#subtitle: There's lots to learn!
gh-repo: uber/causalml
gh-badge: [star, fork, follow]
#tags: [test]
comments: false
#author: Bill Smith
---

# Installation

Installation with `conda` is recommended.

`conda` environment files for Python 3.7, 3.8 and 3.9 are available in the repository. To use models under the `inference.tf` module (e.g. `DragonNet`), additional dependency of `tensorflow` is required. For detailed instructions, see below.

## Install using `conda`:

Install `conda` with:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.0-3-Linux-x86_64.sh
bash Miniconda3-py38_23.5.0-3-Linux-x86_64.sh -b
source miniconda3/bin/activate 
conda init
source ~/.bashrc 
```

## Install from source:

### Create a clean conda environment

```
conda create -n causalml-py38 -y python=3.8
conda activate causalml-py38
conda install -c conda-forge cxx-compiler
conda install python-graphviz
conda install -c conda-forge xorg-libxrender
```

Then:

```bash
git clone https://github.com/uber/causalml.git
cd causalml
pip install .
python setup.py build_ext --inplace
```

# Quick Start

## Average Treatment Effect Estimation with S, T, X, and R Learners

```python
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor
from causalml.dataset import synthetic_data

y, X, treatment, _, _, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
te, lb, ub = nn.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub = xl.estimate_ate(X, treatment, y, e)
print('Average Treatment Effect (BaseXRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

rl = BaseRRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub =  rl.estimate_ate(X=X, p=e, treatment=treatment, y=y)
print('Average Treatment Effect (BaseRRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))
```

See the [Meta-learner example notebook](https://github.com/uber/causalml/blob/master/docs/examples/meta_learners_with_synthetic_data.ipynb) for details.


