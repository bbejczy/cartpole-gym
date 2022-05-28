# cartpole-gym

Repository for homeworks of class PD803.

## List of algorithms:
- Actor Critic without Experience Replay: `AC.py`
- Actor Critic with Experience Replay: `AC_ER.py`
- Dualing Deep Q Network with Experience Replay: `DDQN_ER.py`
- Asynchronous Advantage Actor Critic: `A3C_continuous.py`
- Asynchronous Advantage Actor Critic with Experience Replay: `A3C_ER.py`
- Asynchronous Advantage Actor Critic for Discrete Action Space: `A3C_original.py`
  
## Logging and sweeps

For logging and parameter sweeps [Weights and Biases](https://wandb.ai/site) has been used, you can turn this feature of by passing the argument `--wandb=False` (or `logging = False` in some cases).

Same library has been used for creating the [grid sweeps](https://wandb.ai/bbejczy/cartpole-gym/sweeps) for the DDQN network variations. The configuration can be found in `grid_sweep.yml`. 
