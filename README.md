# varibad_working
Code taken from this repository and adjusted to my requirements: https://github.com/lmzintgraf/varibad/tree/master

# project 
Build a bicameral agent with left/right hemispheres.
Right hemisphere is designed to be a 'generalist' while left will learn specific tasks
We meta-train the right hemisphere using RL^2 (using varibad implementation)
We evaluate the combined bicameral agent in a continual learning context.

We use metaworld for our environment. We meta-train using ML10 training tasks and then evaluate on the ML10 test tasks.

To perform our continual learning evaluation we will train an agent on the ML10 test tasks sequentially, and evaluate test performance on all tasks. Here we can assess forgetting and worst-case performance.

## TODO:

### 1. Set up eval for ML10
We want to evaluate model performance on ML10 environments. We want to be able to do so to verify the performance of RL^2 on ML10 test and train tasks.
Get all historical actions / rewards and pass through model. eval on percent success on each environment in train and test tasks.
Perhaps include this in the meta-learner function?

### 2. Set up continual learning training + evaluation
Set up a training loop which takes one task and trains on it for a specified number of iterations. Evaluate on all environments.
What order of tasks? does it matter? do we want to look at different permutations/orderings?

### 3. Build models to train
After meta-training the right hemisphere, we need:
1. bi-hemispheric network with right + left hemisphere and gating network
2. EWC (single untrained network)
3. EWC (applied to bi-hemispheric)
4. Others should be relatively easy - e.g. left-untrained, right meta-trained, left + right untrained?

### 4. Environment requirements
Need to create environments that can perform continual learning.
Use continual world as an example.
For Meta-world, do I want to randomly sample changing goals / objectives? or do I actually want to use the multi-task equivalent?

#### Questions:
- is ML10 test tasks ok for continual eval? or should I use the multi-task equivalent, which have (I think) common structure?
- How many historical action, reward, obs values do I need to provide for evaluation of RL^2?
- How to evaluate performance on Metaworld tasks? Use percent success? (seems to be what they did in paper)
- training RL^2 on rewards - Metaworld tasks seem to have quite varied reward values. Does this skew the training?
- should metaworld tasks be 'done' on success of task?