We modified this original repo: https://github.com/leggedrobotics/legged_gym, to add training via RM

Original RM code from https://github.com/RodrigoToroIcarte/reward_machines
We modified RM code to work with vectorized environments

# Train

```
python3 legged_gym/scripts/train.py --task=a1_bounding --headless
```

# Test:

    First update load_run and checkpoint in config file. Then, run:

```
python3 legged_gym/scripts/play.py --task=a1_bounding
```

# TODO

Compute next RM states in parallel, to keep all computations on the GPU