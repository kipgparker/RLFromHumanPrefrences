# RLFromHumanPrefrences
Reinforcement learning from human preferences to produce behavior unaligned from environment rewards, learning via human preferences with [Garner](https://garner.app). tools.

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Paper
### Deep reinforcement learning from human preferences
Reproduction of OpenAI and Deepminds project using [Garner](https://garner.app) tools to gather human prefrences efficiently on a large scale.

[[Original Blog]](https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/)
[[Original Paper]](https://arxiv.org/abs/1706.03741)

