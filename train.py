import gymnasium as gym
import numpy as np
from TradingEnv import TradingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecEnv
class CustomTradingCallback(BaseCallback):
    """
    A custom callback for logging additional trading environment information to TensorBoard.

    Logs:
    - Final portfolio value at the end of each episode.
    - Final profit/loss percentage at the end of each episode.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Variables to store episode metrics if needed across steps (not used here currently)
        # self.episode_rewards = []
        # self.episode_lengths = []

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It checks for episode completions and logs final info.

        :return: True to continue training, False to stop.
        """
        # Check if any environment episodes have finished
        # self.locals is a dict containing info from the rollout
        # 'dones' is a boolean array indicating episode termination
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            # Should not happen in standard SB3 training loop
            return True

        for i, done in enumerate(dones):
            if done:
                # Episode finished for environment i
                # Access the info dict for this environment
                info = infos[i]

                # SB3 wrappers like VecMonitor might store final info here
                final_info = info.get("final_info")

                if final_info is not None:
                    # Log metrics from the final_info dictionary
                    final_value = final_info.get('current_total_value')
                    final_pl_pct = final_info.get('profit_loss_pct')

                    if final_value is not None:
                        if self.verbose > 0:
                            print(f"Episode End: Final Value={final_value:.2f}, P/L%={final_pl_pct*100:.2f}%")
                        # Log to TensorBoard (and other loggers)
                        # SB3 convention uses 'rollout/' prefix for episode stats
                        self.logger.record("rollout/ep_final_total_value", final_value)
                        self.logger.record("rollout/ep_final_profit_loss_pct", final_pl_pct)

                # You could also log info directly from 'info' if not using VecMonitor/final_info
                # else:
                #    final_value = info.get('current_total_value')
                #    ... log ...

        # Must return True to continue training
        return True

    # You can also override other methods like _on_training_start, _on_rollout_end, etc.
    # For example, logging hyperparameters at the start:
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            # add other hyperparameters of interest here
        }
        # define the metrics that will appear in the `HPARAMS` dashboard
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
            "rollout/ep_final_total_value": 0.0, # Add custom metrics here too
            "rollout/ep_final_profit_loss_pct": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
        

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * (1 - progress_remaining)

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
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps, tensorboard_log=log_dir, batch_size=batch_size, learning_rate=linear_schedule(3e-4), ent_coef=ent_coef, clip_range=clip_range, gamma=gamma)
model.learn(total_timesteps=3000000, progress_bar=True, tb_log_name="stock_ppo", callback=CustomTradingCallback(), log_interval=1)
model.save("stock_ppo_model_3M")