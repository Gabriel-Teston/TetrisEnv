from gym.envs.registration import register

register(
    id='tetris-v0',
    entry_point='Tetris.envs.tetris_env:TetrisEnv'
)