We modified this original repo: https://github.com/leggedrobotics/legged_gym, to add training via RM

Our RM implementation inspired by original RM code from https://github.com/RodrigoToroIcarte/reward_machines
Our implementation is compatible with vectorized environments on GPU

# Train

Gait types: trot, bound, pace, walk, bound_walk
experiment types: rm, naive, augmented, naive3T, no_gait


Example command to train pace gait, with rm state included in state space, on random seed 18:

```
python3 legged_gym/scripts/train.py --task=a1_rm --gait=pace --experiment=rm --seed=18 --headless
```

-- headless means training will not be visualized.


# Validate:

Sim-to-sim transfer to PyBullet:

```
cd pybullet_val
python3 -m scripts.play_bullet.py
```

# Test:

    First update load_run and checkpoint in config file. Then, run:

```
python3 legged_gym/scripts/play.py --task=a1_rm --gait=pace --experiment=rm --seed=0
```
