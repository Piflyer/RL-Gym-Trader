import gymnasium as gym
import numpy as np
from RSLTradingEnv import TradingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecEnv, VecMonitor # Import VecMonitor
import yfinance as yf
from tqdm import tqdm
import yaml
import requests_cache
import datetime
import pandas as pd

cache_name = 'yfinance_cache'
expire_after = datetime.timedelta(days=1) # Cache expires after 1 day

# Create a cached session
session = requests_cache.CachedSession(
    cache_name=cache_name,
    backend='sqlite',
    expire_after=expire_after
)

class Dataloader:
    def __init__(self, period, session, extra_obs, max_episode_length=1000):
        self.session = session
        self.period = period
        self.extra_obs = extra_obs
        self.max_episode_length = max_episode_length
        self.vix_data = yf.Ticker("^VIX", session=self.session).history(period=period, auto_adjust=True)
        self.gspc_data = yf.Ticker("^GSPC", session=self.session).history(period=period, auto_adjust=True)
        self.random_symbols = [
            "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL", "META", "ADBE",
            "NFLX", "VOO", "FTEC", "TSLA", "JPM", "V", "UNH"
        ]
    
    def dataloader(self): 
        all_stock_data = {}
        for symbol in tqdm(self.random_symbols, desc="Fetching stock data"):
                all_stock_data[symbol] = self.batch_fetch_data(symbol)
        return all_stock_data
    
    def batch_fetch_data(self, stock):
        # print(f"[INFO] Fetching data for {stock}...")
        try:
            stock_ticker = yf.Ticker(stock, session=self.session)
            stock_data = stock_ticker.history(period=self.period, auto_adjust=True)
            if self.extra_obs:
                stock_df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not stock_data.empty else pd.DataFrame(index=stock_data.index)
                vix_df = self.vix_data[['Close']].rename(columns={'Close': 'VIX_Close'}).copy() if not self.vix_data.empty else pd.DataFrame(index=self.vix_data.index)
                gspc_df = self.gspc_data[['Close']].rename(columns={'Close': 'GSPC_Close'}).copy() if not self.gspc_data.empty else pd.DataFrame(index=self.gspc_data.index)
                merged_data = pd.concat([stock_df, vix_df, gspc_df], axis=1, join='outer')
                if 'Close' not in merged_data.columns and not stock_df.empty:
                    print("[ERROR] Primary stock 'Close' column missing after outer join.")
            else:
                stock_df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not stock_data.empty else pd.DataFrame(index=stock_data.index)
                merged_data = stock_df
            merged_data = merged_data.groupby(merged_data.index.date).first()
            merged_data.index = pd.to_datetime(merged_data.index) # Ensure index is datetime
            
            merged_data = merged_data.ffill(limit=3)
            # Handle NaNs in critical columns
            if 'Volume' in merged_data.columns:
                merged_data['Volume'].fillna(0.0, inplace=True) # Fill NaN volume with 0
            merged_data.dropna(subset=['Close'], inplace=True) # Drop rows ONLY if 'Close' is missing
            min_required_length = 35 + self.max_episode_length
            if len(merged_data) < min_required_length:
                print(f"[WARNING] Insufficient merged data after processing for {stock} (Final Length: {len(merged_data)}, Required: {min_required_length}).")
                return None
            
            return merged_data
        except Exception as e:
            print(f"[ERROR] Exception during data fetch/process for {stock}: {e}")
            return None

dataloader = Dataloader(period="max", session=session, extra_obs=True)
dataloader = dataloader.dataloader()

class ConfigParser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

configPasrer = ConfigParser('sb3_config.yaml')

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func

env_kwargs = {
    "dataloader": dataloader,
    "max_episode_length": configPasrer.get("num_steps", 1000),
    "extra_obs": configPasrer.get("extra_obs", True),
    "device": configPasrer.get("device", "cpu"),
    "rl_platform": configPasrer.get("rl_platform", "SB3"),
    "period": configPasrer.get("period", "max"),
    "extra_obs": configPasrer.get("extra_obs", True),
    "use_privileged_obs": configPasrer.get("use_privileged_obs", False),
}


log_dir = "./tensorboard_logs_refactored/"
n_envs = 32
n_steps = 1000
batch_size = 128
ent_coef = 0.02
gamma = 0.9693339928617053
clip_range = 0.2151957180412349
initial_lr = 3e-4

new_logger = configure(log_dir, ["stdout", "tensorboard"])
vec_env = make_vec_env(TradingEnv, n_envs=configPasrer.get("num_envs", 32), env_kwargs=env_kwargs)
vec_env = VecMonitor(vec_env) # Wrap the VecEnv with VecMonitor
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
# model.learn(total_timesteps=configPasrer.get("total_timesteps", 2_000_000), 
#             progress_bar=True, 
#             tb_log_name=configPasrer.get("name", "refactored_stock_ppo_model_2M"), 
#             log_interval=1)
model.learn(total_timesteps=2_000_000, progress_bar=True, tb_log_name="stock_ppo", log_interval=1)

model.save(f"models/{configPasrer.get('name', 'refactored_stock_ppo_model_2M')}")
