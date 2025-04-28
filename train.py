from RSLTradingEnv import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor # Import VecMonitor
import requests_cache
import datetime
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from utils.helpers import *

cache_name = 'yfinance_cache'
expire_after = datetime.timedelta(days=1) # Cache expires after 1 day

# Create a cached session
session = requests_cache.CachedSession(
    cache_name=cache_name,
    backend='sqlite',
    expire_after=expire_after
)

configPasrer = ConfigParser('sb3_config.yaml')
dataloader = Dataloader(period="max", session=session, extra_obs=True, extended=configPasrer.get("extended_data", False))
dataloader = dataloader.dataloader()

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func

if configPasrer.get("curriculum") is not None:
    curriculum_manager = CurriculumStepper(
        step_size=configPasrer.get("curriculum_step_size", 1),
        max_steps=configPasrer.get("curriculum_max_steps", 10),
        curriculum_name=configPasrer.get("curriculum", ["curriculum"])
    )
else: 
    curriculum_manager = None

env_kwargs = {
    "dataloader": dataloader,
    "max_episode_length": configPasrer.get("num_steps", 1000),
    "extra_obs": configPasrer.get("extra_obs", True),
    "device": configPasrer.get("device", "cpu"),
    "rl_platform": configPasrer.get("rl_platform", "SB3"),
    "period": configPasrer.get("period", "max"),
    "extra_obs": configPasrer.get("extra_obs", True),
    "use_privileged_obs": configPasrer.get("use_privileged_obs", False),
    "eval": False,
    "eval_buffer": configPasrer.get("eval_steps", 1000),
    "min_percnt": configPasrer.get("min_percnt", 0.8),
    "num_priv_obs": configPasrer.get("num_priv_obs", 0),
    "curriculum_manager": curriculum_manager,
    "failed_trade_terminate": configPasrer.get("failed_trade_terminate", True),
}

#setup for evaluation
eval_env_kwargs = {
    "dataloader": dataloader,
    "max_episode_length": configPasrer.get("eval_steps", 1000),
    "extra_obs": configPasrer.get("extra_obs", True),
    "device": configPasrer.get("device", "cpu"),
    "rl_platform": configPasrer.get("rl_platform", "SB3"),
    "period": configPasrer.get("period", "max"),
    "use_privileged_obs": False,
    "min_percnt": configPasrer.get("min_percnt", 0.8),
    "eval": True,
    "failed_trade_terminate": configPasrer.get("failed_trade_terminate", True),
}
# Create the evaluation environment
eval_env = make_vec_env(TradingEnv, n_envs=1, env_kwargs=eval_env_kwargs)
# Create the evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f'./logs/evaluation/{configPasrer.get("name", "refactored_stock_ppo_model_2M")}',
    log_path=f'./logs/evaluation/{configPasrer.get("name", "refactored_stock_ppo_model_2M")}',
    eval_freq=configPasrer.get("eval_freq", 100000),
    deterministic=True,
    render=False,
)



log_dir = "./tensorboard_logs_refactored/"
# callback = CustomCallback()

callback = CallbackList([CustomCallback(curriculum_manager=curriculum_manager)])


new_logger = configure(log_dir, ["stdout", "tensorboard"])
vec_env = make_vec_env(TradingEnv, n_envs=configPasrer.get("num_envs", 32), env_kwargs=env_kwargs)
vec_env = VecMonitor(vec_env) # Wrap the VecEnv with VecMonitor
if configPasrer.get("use_privileged_obs"):
    print("[INFO] Using privileged observations.")
    model = PPO(PrivilegedPolicy, 
                vec_env, 
                verbose=1, 
                device=configPasrer.get("device", "cpu"),
                n_steps=configPasrer.get("num_steps", 1000),
                tensorboard_log=configPasrer.get("tensorboard_log", log_dir),
                batch_size=configPasrer.get("batch_size", 128),
                learning_rate=linear_schedule(float(configPasrer.get("lr", 3e-4))), 
                ent_coef=configPasrer.get("ent_coef", 0.02), 
                max_grad_norm=configPasrer.get("max_grad_norm", 0.5),
                clip_range=configPasrer.get("clip_range", 0.2),
                gamma=configPasrer.get("gamma", 0.99),
                policy_kwargs=dict(privileged_obs_dim=configPasrer.get("num_priv_obs")),
                )
else:
    print("Using standard policy without privileged observations.")
    model = PPO("MlpPolicy", 
                vec_env, 
                verbose=1, 
                device=configPasrer.get("device", "cpu"),
                n_steps=configPasrer.get("num_steps", 1000),
                tensorboard_log=configPasrer.get("tensorboard_log", log_dir),
                batch_size=configPasrer.get("batch_size", 128),
                learning_rate=linear_schedule(float(configPasrer.get("lr", 3e-4))), 
                ent_coef=configPasrer.get("ent_coef", 0.02), 
                max_grad_norm=configPasrer.get("max_grad_norm", 0.5),
                clip_range=configPasrer.get("clip_range", 0.2),
                gamma=configPasrer.get("gamma", 0.99),
                )
model.learn(total_timesteps=configPasrer.get("total_timesteps", 2_000_000), 
            progress_bar=True, 
            tb_log_name=configPasrer.get("name", "refactored_stock_ppo_model_2M"), 
            log_interval=1,
            #custom callback and eval callback
            callback=callback,
            )
# model.learn(total_timesteps=2_000_000, progress_bar=True, tb_log_name="stock_ppo", log_interval=1)

model.save(f"models/{configPasrer.get('name', 'refactored_stock_ppo_model_2M')}")

#eval mode:
# obs_actor_only = full_obs[:, :-priv_dim]