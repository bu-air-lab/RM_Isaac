We modified this original repo: https://github.com/leggedrobotics/legged_gym, to add training via RM

Original RM code from https://github.com/RodrigoToroIcarte/reward_machines
We modified RM code to work with vectorized environments

# Train

Gait types: walk, trot, bounce
experiment types: rm, naive, augmented, naive3T

```
python3 legged_gym/scripts/train.py --task=a1_bounding --gait=walk --experiment=rm --headless
```

# Test:

    First update load_run and checkpoint in config file. Then, run:

```
python3 legged_gym/scripts/play.py --task=a1_bounding --gait=walk --experiment=rm
```
