## Attention CNN policy Agent on Procgen Benchmark

### Environment

Procgen consists of 16 unique environments with procedural generation logic governing the level output, selection of game assets, location and spawn times of entities and other game-specific details. All the environments utilize a discrete 15-dimensional action space and produce observations (64*64*3 RGB images) of same specifications. Many environments include non-operational actions, to support a unified training pipeline.
  The environments are optimized to perform thousands of steps per second on a single CPU core, including time to render observations. This enables a fast experimental pipeline. 

### Approach

A trainable attention module for CNN architectures often utilized for image classifcation is used to improve performance over baseline IMPALA models of similar depths. A2C-PPO implementation from [ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) is used to primary base to build on. 

### Train
For attention network:
'''
python train.py --exp_name attention_25m --env_name chaser --start_level 0 --num_levels 200 --param_name easy-custom --device gpu --log_level 30 --num_timesteps 25000000
'''

### References
1. Cobbe, Karl, et al. "Leveraging procedural generation to benchmark reinforcement learning." International
conference on machine learning. PMLR, 2020.
2. Jetley, Saumya, et al. "Learn to pay attention." arXiv preprint arXiv:1804.02391 (2018).
3. https://github.com/joonleesky/train-procgen-pytorch
4. https://github.com/openai/procgen
5. https://github.com/SaoYan/LearnToPayAttention

