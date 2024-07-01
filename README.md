# Right-left brain in a Reinforcement Learning Agent
We create a bi-hemispheric agent with specialised hemispheres based on the Novelty Routine Hypothesis.  

# Running
## Bi-hemispheric agents / left-only baselines
These agents can be run from `run_continuallearner.py`.

Example run Bi-cameral agent:
```python ./run_continual_learner.py --seed 808 --learning_rate 0.00001 --run_name "random_init_fixed20_gatepen_reach-v2" --env_name "reach-v2" --run_folder "rl2_baseline/rl2_bicameral_baseline" --num_mini_batch 8 --ppo_epoch 8 --num_processes 20 --randomization "random_init_fixed20" --algorithm "bicameral" --steps_per_env 5000000 --log_folder "logs/bicameral_net" --entropy_coef 0.00001 --use_gating_penalty True --gating_alpha 0.75 --gating_beta 5```

Example run left-only baseline:
```python ./run_continual_learner.py --seed 808 --run_name "random_init_fixed20_reach-v2" --env_name "door-open-v2" --run_folder "rl2_baseline/rl2_double_baseline" --num_mini_batch 8 --ppo_epoch 8 --num_processes 20 --randomization "random_init_fixed20" --algorithm "left_only" --steps_per_env 5000000 --log_folder "logs/left_only" --entropy_coef 0.00001 --learning_rate 0.00001```

## Random / right-only baselines
As Random / Right-only baselines are just evaluated on sampled tasks, they run much faster and can be done in batches using `run_scenarios.py`

Example run Random agents:
```python ./run_scenarios.py --algorithm "random" --steps_per_env 100000 --log_folder ./random_agent --randomization "random_init_fixed20" --num_processes 20```


## Meta-learning
We have saved the trained models in `rl2_baseline`.
- rl2_bicameral_baseline: the right-hemisphere network
- rl2_double_baseline: the right-only baseline
- rl2_bicameral_small_baseline: a small baseline used during development

To generate your own baseline you can run the `main.py` script, taken from the VariBad repository (see sources)

## Sources
Meta-training code for RL2 is taken from here: https://github.com/lmzintgraf/varibad/tree/master
We also directly copied some environment utilities from the Continual World repo here: https://github.com/awarelab/continual_world