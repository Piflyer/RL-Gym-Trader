import gymnasium as gym
import numpy as np
# from TradingEnv_old import TradingEnv
from RSLTradingEnv import TradingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecEnv, VecMonitor # Import VecMonitor

        

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func

log_dir = "./tensorboard_logs_refactored/"
n_envs = 16
n_steps = 1000
batch_size = 128
ent_coef = 0.02
gamma = 0.9693339928617053
clip_range = 0.2151957180412349
new_logger = configure(log_dir, ["stdout", "tensorboard"])
vec_env = make_vec_env(TradingEnv, n_envs=n_envs)
initial_lr = 3e-4
vec_env = VecMonitor(vec_env) # Wrap the VecEnv with VecMonitor
ent_coef = 0.02  # Default: 0.0 (increase to encourage exploration)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps, tensorboard_log=log_dir, batch_size=batch_size, learning_rate=linear_schedule(3e-4), ent_coef=ent_coef, clip_range=clip_range, gamma=gamma)
model.learn(total_timesteps=2_000_000, progress_bar=True, tb_log_name="stock_ppo", log_interval=1)
model.save("refactored_stock_ppo_model_2M")
