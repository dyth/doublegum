# DoubleGum

Code for *Double Gumbel Q-Learning*

## Installation

On `Python 3.9` with `Cuda 12.2.1` and `cudnn 8.8.0`.

```commandline
git clone git@github.com:dyth/doublegum.git
cd doublegum
```

create virtualenv
```
virtualenv <VIRTUALENV_LOCATION>/doublegum
source <VIRTUALENV_LOCATION>/doublegum
```
or conda
```commandline
conda create --name doublegum python=3.9
conda activate doublegum
```

install mujoco
```commandline
mkdir .mujoco
cd .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```

install packages
```commandline
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

test that the code runs
```commandline
./test.sh
```

## Continuous Control

```commandline
main_cont.py --env <ENV_NAME> --policy <POLICY>
```
MetaWorld `env`s are run with `--env MetaWorld_<ENVNAME>`

Policies benchmarked in our paper were:
* `DoubleGum`: DoubleGum (our algorithm)
* `DDPG`: DDPG (Deep Deterministic Policy Gradients), [[Lilicrap et al., 2015](https://arxiv.org/abs/1509.02971)]
* `TD3`: TD3 (Twin Delayed DDPG), [[Fujimoto et al., 2018](https://proceedings.mlr.press/v80/fujimoto18a.html)]
* `SAC`: SAC (Soft Actor Critic, defaults to use Twin Critics), [[Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)]
* `XQL --ensemble 1`: XQL (Extreme Q-Learning), [[Garg et al., 2023](https://openreview.net/forum?id=SJ0Lde3tRL)]
* `MoG-DDPG`: MoG-DDPG (Mixture of Gaussians Critics DDPG), [[Barth-Maron et al., 2018](https://openreview.net/forum?id=SyZipzbCb), [Shariari et al, 2022](https://arxiv.org/abs/2204.10256)]

Policies we created/modified as additional benchmarks were:
* `QR-DDPG`: QR-DDPG (Quantile Regression [[Dabney et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11791)] with DDPG, defaults to use Twin Critics)
* `QR-DDPG --ensemble 1`: QR-DDPG without Twin Critics
* `SAC --ensemble 1`: SAC without Twin Critics
* `XQL`: XQL with Twin Critics
* `TD3 --ensemble 5 --pessimism <p>`: Finer TD3, where p is an integer between 0 and 4

Policies included in this repository but not benchmarked in our paper were:
* `IQL`: Implicit Q-Learning adapted to an online setting, [[Kostrikov et al., 2022](https://openreview.net/forum?id=68n2s9ZJWF8)]
* `SACLite`: SAC without the entropy term on the critic, [[Yu et al., 2022](https://arxiv.org/abs/2201.12434)]

## Discrete Control

```commandline
main_disc.py --env <ENV_NAME> --policy <POLICY>
```

Policies benchmarked in our paper were:
* `DoubleGum`: DoubleGum (our algorithm)
* `DQN`: DQN, [[Mnih et al., 2015](https://www.nature.com/articles/nature14236)]
* `DDQN`: DDQN (Double DQN), [[van Hasselt et al., 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10295)]
* `DuellingDQN`: DuellingDQN, [[Wang et al., 2016](http://proceedings.mlr.press/v48/wangf16.html)]

Policies we created/modified as additional benchmarks were:
* `DuellingDDQN`: DuellingDDQN (Duelling Double DQN)


## Acknowledgements

* Wrappers from [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl)
* Distributional RL from [google-deepmind/acme](https://github.com/google-deepmind/acme)
* Control flow from [yifan12wu/td3-jax]([td3-jax](https://github.com/yifan12wu/td3-jax)
