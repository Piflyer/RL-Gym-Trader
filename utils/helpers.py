import yfinance as yf
from tqdm import tqdm
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import yaml
import numpy as np
import pandas as pd
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import torch
from gymnasium.spaces import Box
from collections import deque


class CurriculumStepper:
    def __init__(self, step_size=1, max_steps=10, curriculum_name=["curriculum"]):
        self.step_size = step_size
        self.max_steps = max_steps
        self.curriculum_name = curriculum_name
        self.curriculums = {symbol: 0 for symbol in self.curriculum_name}
    
    def step(self, symbol):
        if symbol not in self.curriculum_name:
            raise ValueError(f"Symbol {symbol} not in curriculum.")
        # Check if the current step is less than the max steps
        # and increment the step size
        if self.curriculums[symbol] < self.max_steps:
            self.curriculums[symbol] += self.step_size
            return True
        else:
            return False
    def reset(self):
        self.curriculums = {symbol: 0 for symbol in self.curriculum_name}
        return True
    def get_curriculum(self, symbol):
        if symbol not in self.curriculum_name:
            return None
        return self.curriculums[symbol]
    def get_max_steps(self):
        return self.max_steps

class PadPrivilegedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, priv_dim: int = 18):
        super().__init__(env)
        self.priv_dim = priv_dim

        old = env.observation_space
        # new shape = (old_dim + priv_dim, 1)
        new_shape = (old.shape[0] + self.priv_dim, *old.shape[1:])

        # create bounds of -inf / +inf for **all** dims
        low  = np.full(new_shape, -np.inf, dtype=old.dtype)
        high = np.full(new_shape,  np.inf, dtype=old.dtype)

        self.observation_space = Box(low=low, high=high, dtype=old.dtype)

    def observation(self, obs):
        # obs comes in shape (50,1); flatten or keep dims
        base = np.array(obs)
        # pad the **data** with zeros (or whatever stand-in you choose)
        pad  = np.zeros((self.priv_dim, *base.shape[1:]), dtype=base.dtype)
        return np.concatenate([base, pad], axis=0)


class CustomCallback(BaseCallback):
    """
    Callback for logging average extras['net_gain'],
    extras['holding_gains'], and extras['current_networth']
    to TensorBoard at the end of each rollout.
    """
    def __init__(self, verbose=0, curriculum_manager=None):
        super().__init__(verbose)
        # buffers to accumulate values during rollout
        self.net_gain_buffer = []
        self.holding_gains_buffer = []
        self.current_networth_buffer = []
        self.curriculum_manager = curriculum_manager
        self.curriculum_net_gains = deque(maxlen=100)
        self.curriculum_holding_gains = deque(maxlen=100)
        self.curriculum_rewards = deque(maxlen=100)
        self.reward_buffer = []

    def _on_step(self) -> bool:
        # infos is a list of info-dicts, one per parallel env
        infos = self.locals.get("infos", None)
        self.reward_buffer.append(np.mean(self.locals["rewards"]))
        if infos is not None:
            for info in infos:
                if "net_gain" in info:
                    self.net_gain_buffer.append(info["net_gain"])
                if "holding_gains" in info:
                    self.holding_gains_buffer.append(info["holding_gains"])
                if "current_networth" in info:
                    self.current_networth_buffer.append(info["current_networth"])
        return True

    def _on_rollout_end(self) -> None:
        # compute averages if we have any data
        if len(self.net_gain_buffer) > 0:
            avg_net = np.mean(self.net_gain_buffer)
            avg_hold = np.mean(self.holding_gains_buffer)
            avg_nw  = np.mean(self.current_networth_buffer)

            # record to TB under "train/..."
            self.logger.record("train/avg_net_gain",        avg_net)
            self.logger.record("train/avg_holding_gains",   avg_hold)
            self.logger.record("train/avg_current_networth", avg_nw)

        # clear for next rollout
        self.curriculum_net_gains.append(np.mean(self.net_gain_buffer))
        self.curriculum_holding_gains.append(np.mean(self.holding_gains_buffer))
        self.curriculum_rewards.append(np.sum(self.locals["rewards"]))
        self.net_gain_buffer.clear()
        self.reward_buffer.clear()
        self.holding_gains_buffer.clear()
        self.current_networth_buffer.clear()
        if self.curriculum_manager is not None:
            self.logger.record("curriculum/current_greedy_reward_step", self.curriculum_manager.get_curriculum("greedy_reward"))
            if len(self.curriculum_net_gains) > 0:
                avg_net_gain = np.mean(self.curriculum_net_gains)
                avg_holding_gain = np.mean(self.curriculum_holding_gains)
                self.logger.record("curriculum/avg_net_gain", avg_net_gain)
                self.logger.record("curriculum/avg_holding_gain", avg_holding_gain)
                std_net_gain = np.std(self.curriculum_net_gains)
                std_holding_gain = np.std(self.curriculum_holding_gains)
                self.logger.record("curriculum/std_net_gain", std_net_gain)
                self.logger.record("curriculum/std_holding_gain", std_holding_gain)
                std_rewards = np.std(self.curriculum_rewards)
                avg_rewards = np.mean(self.curriculum_rewards)
                self.logger.record("curriculum/std_rewards", std_rewards)
                # check standard deviation
                if len(self.curriculum_net_gains) == 100:
                    if (avg_net_gain > 0.13 and std_net_gain < 0.45) or (avg_rewards > 40 and std_rewards < 4):
                        self.curriculum_manager.step("greedy_reward")
                        print(f"[INFO] Curriculum step for greedy_reward: {self.curriculum_manager.get_curriculum('greedy_reward')}")
                        self.curriculum_net_gains.clear()
                        self.curriculum_holding_gains.clear()

class PrivilegedPolicy(ActorCriticPolicy):
    def __init__(self, *args, privileged_obs_dim=0, **kwargs):
        self.privileged_obs_dim = privileged_obs_dim
        super(PrivilegedPolicy, self).__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self):
        # Build the MLP extractor with the privileged observation dimension
        self.actor_net = torch.nn.Sequential(
            torch.nn.Linear(self.features_dim - self.privileged_obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(self.features_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        
        class PrivilegedExtractor(torch.nn.Module):
            def __init__(self, actor_net, critic_net, privileged_obs_dim):
                super().__init__()
                self.actor_net = actor_net
                self.critic_net = critic_net
                self.privileged_obs_dim = privileged_obs_dim
                self.latent_dim_pi = 128  # Final output size of actor_net
                self.latent_dim_vf = 128  

            def forward(self, features):
                # SB3 requires this, but you don't have to use it directly
                return self.forward_actor(features), self.forward_critic(features)

            def forward_actor(self, features):
                actor_input = features[:, :-self.privileged_obs_dim]
                return self.actor_net(actor_input)

            def forward_critic(self, features):
                return self.critic_net(features)
            
        self.mlp_extractor = PrivilegedExtractor(self.actor_net, self.critic_net, self.privileged_obs_dim)

class Dataloader:
    def __init__(self, period, session, extra_obs, max_episode_length=1000, extended=False):
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
        
        self.random_symbols_extend = [
            # Communication Services
            "T", "VZ", "META", "CMCSA",
            # Consumer Discretionary
            "AMZN", "TSLA", "NKE", "MCD", "HD",
            # Consumer Staples
            "PG", "KO", "PEP", "WMT",
            # Energy
            "XOM", "CVX", "COP", "SLB", "OXY",
            # Financials
            "JPM", "BAC", "WFC", "GS", "C",
            # Healthcare
            "JNJ", "PFE", "MRK", "UNH", "ABT",
            # Industrials
            "BA", "CAT", "HON", "MMM", "UNP",
            # Information Technology
            "AAPL", "MSFT", "GOOGL", "NVDA", "ADBE",
            # Materials
            "SHW", "DD", "LIN", "NEM",
            # Real Estate
            "AMT", "PLD", "SPG", "AVB",
            # Utilities
            "NEE", "DUK", "SO", "D"
        ]
        
        if extended:
            self.symbols = self.random_symbols_extend
        else:
            self.symbols = self.random_symbols
    
    def dataloader(self): 
        all_stock_data = {}
        print("[INFO] Fetching data...")
        for symbol in tqdm(self.symbols, desc="Fetching stock data"):
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