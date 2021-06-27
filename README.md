## Attention CNN policy Agent on Procgen Benchmark

### Environment

Procgen consists of 16 unique environments with procedural generation logic governing the level output, selection of game assets, location and spawn times of entities and other game-specific details. All the environments utilize a discrete 15-dimensional action space and produce observations (64*64*3 RGB images) of same specifications. Many environments include non-operational actions, to support a unified training pipeline.
  The environments are optimized to perform thousands of steps per second on a single CPU core, including time to render observations. This enables a fast experimental pipeline. 

### Approach

