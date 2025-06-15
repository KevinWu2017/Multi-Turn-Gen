# Multi-Turn-Gen
This is a repo based on [caesar](https://github.com/ScalingIntelligence/caesar). This repo is mainly recontructing the original repo, and change some control logics.

To use this repo:
1. First clone this repo and prepare a virtual env. Then run:
```shell
git submodule update --init --recursive
cd third_party/KernelBench
pip install -e .
```

2. Desinate the server and port and api key.

3. Run the script:
```shell
python run.py
```