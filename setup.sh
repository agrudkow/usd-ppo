#!/bin/bash

#prepare venv
python -m venv usd-ppo
source usd-ppo/bin/activate

#install spinup
git clone https://github.com/openai/spinningup.git
git checkout 038665d62d569055401d91856abb287263096178
cd spinningup
pip install -e .
cd ../

#install ray rllib
pip install "ray[rllib]" tensorflow torch

#install mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzvf ./mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mv ./mujoco210 ~/.mujoco
pip install 'mujoco-py<2.2,>=2.1'

#install Jupyter notebook
pip install notebook~=6.4.7