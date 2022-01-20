#!/bin/bash

#prepare venv
python3 -m venv usd-ppo
source usd-ppo/bin/activate

#install spinup
git clone https://github.com/openai/spinningup.git
cd spinningup
git checkout 038665d62d569055401d91856abb287263096178
pip install -e .
cd ../

#install mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzvf ./mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mv ./mujoco210 ~/.mujoco
pip install 'mujoco-py<2.2,>=2.1'

# Install requirements
pip install -r requirements.txt
