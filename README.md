We modified this original repo: https://github.com/leggedrobotics/legged_gym, to add training via RM

Original RM code from https://github.com/RodrigoToroIcarte/reward_machines
We modified RM code to work with vectorized environments

# Train

Gait types: trot, bound, pace, walk, bound_walk
experiment types: rm, naive, augmented, naive3T, no_gait


Example command to train pace gait, with rm state included in state space, on random seed 18:

```
python3 legged_gym/scripts/train.py --task=a1_bounding --gait=pace --experiment=rm --seed=18 --headless
```

-- headless means training will not be visualized.

# Test:

    First update load_run and checkpoint in config file. Then, run:

```
python3 legged_gym/scripts/play.py --task=a1_bounding --gait=pace --experiment=rm
```
