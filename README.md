# cartpole-gym

Repository for homeworks of class PD803.

## List of algorithms:
- Actor Critic without Experience Replay: `AC.py`
- Actor Critic with Experience Replay: `AC_ER.py`
- Dualing Deep Q Network: `DDQN.py`

## Logging and sweeps

For logging and parameter sweeps [Weights and Biases](https://wandb.ai/site) has been used, you can turn this feature of by passing the argument `--wandb=False`.

Same library has been used for creating the [grid sweeps](https://wandb.ai/bbejczy/cartpole-gym/sweeps) for the DDQN network variations. The configuration can be found in `grid_sweep.yml`. 
