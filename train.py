import gymnasium as gym
from TradingEnv import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from torch.optim import Adam

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log custom metrics across all environments
        for i in range(len(self.locals["infos"])):
            if "current_networth" in self.locals["infos"][i]:
                self.logger.record(f"networth/env_{i}", self.locals["infos"][i]["current_networth"])
            if "current_gain" in self.locals["infos"][i]:
                self.logger.record(f"gain/env_{i}", self.locals["infos"][i]["diff_norm"])
            
            #log reward for all envs in each step
            if "episode" in self.locals["infos"][i]:
                episode_reward = self.locals["infos"][i]["episode"]["r"]
                self.episode_rewards.append(episode_reward)
                self.logger.record(f"reward/env_{i}", episode_reward)
        #log average reward for all envs in each step
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            self.logger.record("reward/average", avg_reward)
            
        # log average gain
        if "current_gain" in self.locals["infos"][0]:
            avg_gain = np.mean([info["diff_norm"] for info in self.locals["infos"]])
            self.logger.record("gain/average", avg_gain)
            
        return True

        # Log PPO losses
    def _on_training_step(self) -> bool:
        loss_info = self.locals["loss"] if "loss" in self.locals else {}
        self.logger.record("loss/policy_loss", loss_info.get("policy_loss", 0))
        self.logger.record("loss/value_loss", loss_info.get("value_loss", 0))
        self.logger.record("loss/entropy_loss", loss_info.get("entropy_loss", 0))

def exponential_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * np.exp(-0.1 * (1 - progress_remaining))
    return func

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * (1 - progress_remaining)
    return func

def adaptive_ent_coef(progress_remaining):
    return 0.02 * (1 - progress_remaining)

print("Loading Env...")

log_dir = "./tensorboard_logs/"
n_envs = 10
n_steps = 1000
batch_size = 128
ent_coef = 0.02
gamma = 0.9693339928617053
clip_range = 0.2151957180412349
new_logger = configure(log_dir, ["stdout", "tensorboard"])
vec_env = make_vec_env(TradingEnv, n_envs=n_envs)
ent_coef = 0.02  # Default: 0.0 (increase to encourage exploration)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps, tensorboard_log=log_dir, batch_size=batch_size, learning_rate=linear_schedule(3e-4), ent_coef=ent_coef, optimizer_class=Adam, clip_range=clip_range, gamma=gamma)
model.learn(total_timesteps=3000000, progress_bar=True, tb_log_name="stock_ppo", callback=TensorboardCallback(), log_interval=1)
model.save("stock_ppo_model_30m")