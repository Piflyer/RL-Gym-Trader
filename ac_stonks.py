import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from torch.optim import Adam
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import warnings
warnings.filterwarnings('ignore')

class PrivilegedNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        """
        feature_dim: total observation dimension (should be 41)
        """
        super().__init__()
        self.non_priv_obs = 36  # first 36 features are non-privileged
        # Actor network processes only non-privileged features
        self.actor_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Critic network processes the full observation (all 41 features)
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Flatten() # Add Flatten layer here
        )
        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

    def forward(self, obs: th.Tensor):
        # obs is expected to be of shape (batch_size, 41)
        latent_pi = self.actor_net(obs[:, :self.non_priv_obs])
        latent_vf = self.critic_net(obs)
        return latent_pi, latent_vf

class PrivilegedStockPolicyNet(ActorCriticPolicy):
    """
    Custom policy that uses privileged observations:
    - The actor gets the first 36 features.
    - The critic gets all 41 features.
    """
    def __init__(self, observation_space: spaces.Box, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, **kwargs):
        # Pass an empty net_arch to disable the default MLP extractor.
        super().__init__(observation_space, action_space, lr_schedule, net_arch={}, activation_fn=activation_fn, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Build and assign our custom network as the mlp_extractor.
        self.mlp_extractor = PrivilegedNetwork(self.observation_space.shape[0])

class GlobalDifficulty:
    def __init__(self, difficulty=0):
        self.difficulty = difficulty
    def increase_difficulty(self):
        self.difficulty += 1
    def decrease_difficulty(self):
        self.difficulty = max(0, self.difficulty - 1)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, global_difficulty=None):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.avg_rewards = []
        self.avg_reward = 0
        self.global_difficulty = global_difficulty
        

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
            self.avg_reward = np.mean(self.episode_rewards)
            if len(self.avg_rewards) < 10000:
                self.avg_rewards.append(self.avg_reward)
            else:
                self.avg_rewards = self.avg_rewards[1:] + [self.avg_reward]
            self.logger.record("reward/average", self.avg_reward)
            
        # log average gain
        if "current_gain" in self.locals["infos"][0]:
            avg_gain = np.mean([info["diff_norm"] for info in self.locals["infos"]])
            self.logger.record("gain/average", avg_gain)
            
        return True

        # Log PPO losses
    def _on_training_step(self):
        loss_info = self.locals["loss"] if "loss" in self.locals else {}
        self.logger.record("loss/policy_loss", loss_info.get("policy_loss", 0))
        self.logger.record("loss/value_loss", loss_info.get("value_loss", 0))
        self.logger.record("loss/entropy_loss", loss_info.get("entropy_loss", 0))
    
    def _on_rollout_end(self) -> bool:
        
        if len(self.avg_rewards) == 10000:
            #check if the average reward is greater than 250 and average variance is greater than 0.25
            if sum(self.avg_rewards) / len(self.avg_rewards) > 250 and sum(self.avg_loss) / len(self.avg_loss) > 0.25:
                self.global_difficulty.increase_difficulty()
                self.avg_rewards = []
            
            elif max(self.avg_rewards) - min(self.avg_rewards) < 50:
                self.global_difficulty.decrease_difficulty()
                self.avg_rewards = []
class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size=0, boughtat="2023-12-01", ticker="AAPL", shares=3, min_trade=10, training=True, privileged=True, adaptive=True, ticker_start_end=None, global_difficulty=GlobalDifficulty(), verbose = False):
        super(StockTradingEnv, self).__init__()
        """_summary_

        Args:
            data (dict): dictionary containing the stock data
            window_size (int): size of the window to use for the observation space
            boughtat (str): date the stock was bought
            ticker (str): ticker of the stock
            shares (int): number of shares bought
            min_trade (int): minimum trade size
            training (bool): whether the environment is in training mode
            privileged (bool): whether the environment is in privileged mode
            adaptive (bool): whether the environment is in adaptive mode
        """
        
        self.verbose = verbose
        self.data = None
        self.window_size = window_size
        self.ticker = ticker
        self.shares = shares
        self.min_trade = min_trade
        self.training = training
        self.privileged = privileged
        self.adaptive = adaptive
        self.idx = 0
        self.untouched_data = data
        
        self.shares = shares
        self.init_networth = 0
        self.ticker_start_end = ticker_start_end
        self.boughtat = boughtat
        self.global_difficulty = global_difficulty
        self.position = True
        
        if self.global_difficulty is None and self.adaptive:
            raise ValueError("Global difficulty must be provided for adaptive mode")
        
        if self.ticker_start_end is None:
            self._ticker_init()
        if self.training:
            self._volatility_index()
            self._training_init()
        else:
            try:
                self.idx = self.untouched_data.index.get_loc(self.boughtat)
                self.init_networth = self.untouched_data["Close"][self.ticker][self.idx] * self.shares
                
                #changing self.data to be only data of that ticker
                start, end = self.ticker_start_end[self.ticker]
                start = (self.idx - 30)
                self.data = self.untouched_data["Close"][self.ticker].iloc[start:end]
                self.idx = 30
            except:
                raise ValueError("Invalid date")
        
        #Buy, Sell, Hold
        self.action_space = gym.spaces.Discrete(2)
        self.networth_rwd = 0
        self.networth = self.init_networth
        self.current_action = 0
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 36), dtype=np.float32
        )
        if self.privileged:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 41), dtype=np.float32
        )
        self._init_obs(True)

    def _ticker_init(self):
        self.ticker_start_end = {}
        for ticker in self.untouched_data.columns.levels[1]:
            ticker_data = self.untouched_data["Close"][ticker]
            #use numeric index for start and end that is not NAN and that I can use iloc
            start = ticker_data.first_valid_index()
            end = ticker_data.last_valid_index()
            #get iloc index
            start = ticker_data.index.get_loc(start)
            end = ticker_data.index.get_loc(end)
            self.ticker_start_end[ticker] = (start, end)
    
    def _volatility_index(self):
        self.volatility_index = {}
        for ticker, (start, end) in self.ticker_start_end.items():
            ticker_data = self.untouched_data["Close"][ticker]
            daily_returns = ticker_data.pct_change().dropna()
            self.volatility_index[ticker] = daily_returns.std()

        #have all of the volatility indexes in a normal distribution
        self.volatility_index = pd.Series(self.volatility_index)
        self.volatility_index = (self.volatility_index - self.volatility_index.mean()) / self.volatility_index.std()
        self.volatility_abs = self.volatility_index.abs()
        self.volatility_index_bins = pd.cut(self.volatility_abs, 10, labels=False)
        self.volatility_index_dict = {}
        for ticker, bin in self.volatility_index_bins.items():
            if bin not in self.volatility_index_dict:
                self.volatility_index_dict[bin] = []
            self.volatility_index_dict[bin].append(ticker)

    
    def _training_init(self):
        #get random ticker from the volatility_index_dict from the bin 0 to the global difficulty
        def get_random_ticker():
            if self.adaptive:
                if self.global_difficulty.difficulty <= 11:
                    self.ticker = np.random.choice(self.untouched_data.columns.levels[1])
                else:
                    self.ticker = np.random.choice(self.volatility_index_dict[self.global_difficulty.difficulty])
                    self.min_trade = np.random.randint(1, self.global_difficulty.difficulty*3)
            else:
                self.ticker = np.random.choice(self.untouched_data.columns.levels[1])
            self.shares = np.random.randint(1, 10)
            self.idx = np.random.randint(self.ticker_start_end[self.ticker][0] + 30, self.ticker_start_end[self.ticker][1])
            self.init_networth = self.untouched_data["Close"][self.ticker][self.idx] * self.shares
        
        while self.idx - 30 <= self.ticker_start_end[self.ticker][0]:
            get_random_ticker()
        start = (self.idx - 30)
        self.data = self.untouched_data["Close"][self.ticker].iloc[start:end]
        self.idx = 30
        
    def _calculate_momentum(self, days):
        if self.idx - days < 0:
            return 0
        past_price = self.data[self.idx - days:self.idx]
        return (past_price[-1] - past_price[0]) / past_price[0]
    
    def _calculate_volatility(self, days):
        if self.idx - days <= 0:
            return 0
        past_price = self.data[self.idx - days:self.idx]
        return past_price.std()
    
    def _init_obs(self, reset=False):
        #observation - price for past 30 days, shares, has postion, weekly and monthly momentum, weekly and monthly volatility, and future 5 days
        self.prev_30 = self.data.iloc[self.idx - 30:self.idx].values.flatten()
        self.m7 = np.asarray(self._calculate_momentum(7)).flatten()
        self.m30 = np.asarray(self._calculate_momentum(30)).flatten()
        self.v7 = np.asarray(self._calculate_volatility(7)).flatten()
        self.v30 = np.asarray(self._calculate_volatility(30)).flatten()
        #self.has_position
        # obs_space = 38
        shares = np.asarray(self.shares).flatten()
        postion = np.asarray(self.position).flatten()
        if self.privileged:
            self.future_5 = self.data[self.idx:self.idx + 5].values.flatten()

            self.obs = np.concatenate([self.prev_30, shares, postion, self.m7, self.m30, self.v7, self.v30, self.future_5])
            #obs_space = 43
        
        else:
            self.obs = np.concatenate([self.prev_30, shares, postion, self.m7, self.m30, self.v7, self.v30])
        
        self.obs = np.expand_dims(self.obs, axis=0).astype(np.float32)
        if reset:
            #non observation space
            if self.training:
                self._training_init()
            self.boughtprice = self.data[self.idx]
            self.streak = 0
            self.current_networth = self.init_networth
            self.max_rel_networth = 0
            self.current_gain = 0
            self.relboughtat = self.idx
            self.networth = self.init_networth
            self.init = True
            self.current_action = 0

        
        
    def reset(self, seed=0):
        super(StockTradingEnv, self).reset()
        if self.training:
            self._training_init()
        else:
            self.idx = self.data.index.get_loc(self.boughtat)
            self.init_networth = self.data[self.boughtat] * self.shares
        self._init_obs(True)
        info = {
            "init_networth" : self.init_networth,
            "current_networth" : self.current_networth,
            "current_action"  : self.current_action,
            "diff": self.current_networth - self.init_networth,
            "diff_norm": (self.current_networth - self.init_networth) / self.init_networth,
            "reward": self._reward(self.current_action),
            "diffculuty": self.global_difficulty.difficulty
        }
        self.position = True
        return self.obs, info
    
    def step(self, action):
        self.init = False
        terminated = False
        truncated = False
        if action == 1:
            if self.streak < self.min_trade:
                terminated = True
            if self.position:
                self.current_networth = self.shares * (self.data[self.idx] - self.boughtprice)
                self.position = False
                self.networth += self.current_networth
                self.current_gain = self.networth - self.init_networth
            else:
                self.current_networth = 0
                self.current_gain = 0
                self.position = True
                if self.networth < self.shares*self.data[self.idx]:
                    terminated = True
            self.streak = 0
            self.relboughtat = self.idx
            self.max_rel_networth = 0
            self.current_action = 1
            self.boughtprice = self.data[self.idx]
        else:
            self.streak += 1
            self.current_action = 0
            self.current_networth = self.shares * (self.data[self.idx] - self.boughtprice)
            if self.current_networth > self.max_rel_networth:
                self.max_rel_networth = self.current_networth
        
            
        
        self.idx += 1
        reward = self._reward(action)
        info = {
            "init_networth" : self.init_networth,
            "current_networth" : self.networth,
            "current_action"  : self.current_action,
            "current_gain" : self.current_networth,
            "diff": self.current_gain,
            "diff_norm": self.current_gain / self.init_networth,
            "reward": self._reward(self.current_action),
            "difficulty": self.global_difficulty.difficulty
        }
        if self.verbose:
            print(" ")
            print("---------------------------------------------------------")
            print(f"[INFO] Current Stock: {self.ticker}")
            print(f"[INFO] Current Day: {self.data.index[self.idx]}")
            print(f"[INFO] Current Open: {self.data[self.idx]}")
            print(f"[INFO] Current Net Worth: {self.networth}")
            print(f"[INFO] Holding Share: {self.position}")
            print(f"[INFO] Current Share Net Worth: {self.current_networth}")
            print("[INFO] Current Action: " + ("Hold" if self.current_action == 0 else "Sell" if self.current_action == 1 and not self.position else "Buy"))
            print("---------------------------------------------------------")
        
        if self.networth < 0.9 * self.init_networth:
            terminated = True
            truncated = False
            return self.obs, reward, terminated, truncated, info
        
        try:
            self._init_obs(False)
        except:
            print("[INFO] End of data")
            terminated = False
            truncated = True
            return self.obs, reward, terminated, truncated, info

        return self.obs, reward, terminated, truncated, info
        
    def _reward(self, action):
        self.reward_weights= {
            "relgain" : 0.7,
            "netgain" : 0.3,
            "streak" : 0.0,
        }
        
        self._rwd_relgain, self._rwd_netgain, self._rwd_streak = 0, 0, 0
        
        if self.position:
            if self.current_networth < self.max_rel_networth and self.max_rel_networth > 0:
                self._rwd_relgain = (self.current_networth - self.max_rel_networth) / self.max_rel_networth
            else:
                self._rwd_relgain = self.current_networth / self.boughtprice
        else:
            self._rwd_relgain = (self.current_networth / self.boughtprice) * -1
        
        self._rwd_netgain = self.current_gain / self.init_networth
        
        if (self.streak < self.min_trade and not self.init) and action == 1:
            self._rwd_streak = -10
        else:
            self._rwd_streak = 2
        
        reward_min = 7
        reward_max = 7
        
        reward = self.reward_weights["relgain"] * self._rwd_relgain + self.reward_weights["netgain"] * self._rwd_netgain + self.reward_weights["streak"] * self._rwd_streak
        reward = np.clip(reward, reward_min, reward_max)
        return reward


def exponential_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * np.exp(-0.1 * (1 - progress_remaining))
    return func

def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func

# def adaptive_ent_coef(progress_remaining):
#     # Ensure that progress_remaining is a scalar or a numpy array
#     if isinstance(progress_remaining, torch.Tensor):
#         progress_remaining = progress_remaining.item()  # Convert to a scalar
    
#     return 0.02 * progress_remaining


### Main Setup
stocks = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMD",  # Technology
    "JNJ", "PFE", "UNH", "MRNA", "LLY",  # Healthcare
    "JPM", "BAC", "GS", "WFC", "C",  # Finance
    "XOM", "CVX", "COP", "SLB", "BP",  # Energy
    "TSLA", "AMZN", "HD", "NKE", "SBUX",  # Consumer Discretionary
    "PG", "KO", "PEP", "WMT", "MO",  # Consumer Staples
    "BA", "CAT", "GE", "UNP", "LMT",  # Industrials
    "NEE", "DUK", "SO", "D", "AEP",  # Utilities
    "PLD", "AMT", "SPG", "O", "VICI",  # Real Estate
    "LIN", "BHP", "SHW", "FCX", "NEM",  # Materials
    "INTC", "IBM", "QCOM", "TXN", "MU",  # Semiconductors
    "NFLX", "DIS", "CMCSA", "T", "VZ",  # Communication Services
    "MMM", "HON", "GD", "RTX",  # Aerospace & Defense
    "ADBE", "CRM", "ORCL", "NOW", "ZM",  # Software
    "COST", "TGT", "DG", "DLTR",  # Retail
    "LOW", "HD", "TSCO", "AZO", "ORLY",  # Home Improvement
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",  # Indexes
]

print("Downloading data...")
# data = yf.download(stocks, period='max')
# data.to_csv("data.csv")
data = pd.read_csv("data.csv", header=[0, 1], index_col=0)

print("Data downloaded, processing...")
#create dictionary to store start and end indexes for each ticker
ticker_start_end = {}
for ticker in data.columns.levels[1]:
    ticker_data = data["Close"][ticker]
    #use numeric index for start and end that is not NAN and that I can use iloc
    start = ticker_data.first_valid_index()
    end = ticker_data.last_valid_index()
    #get iloc index
    start = ticker_data.index.get_loc(start)
    end = ticker_data.index.get_loc(end)
    ticker_start_end[ticker] = (start, end)

print("Intializing environment...")
log_dir = "./tensorboard_logs_test/"
n_envs = 100
n_steps = 1000
batch_size = 1024
ent_coef = 0.02
gamma = 0.85
clip_range = 0.15
new_logger = configure(log_dir, ["stdout", "tensorboard"])
vec_env = make_vec_env(StockTradingEnv, n_envs=n_envs, env_kwargs={
	'data': data,
	'window_size': 10,
	'training': True,
	'privileged': False,
	'adaptive': False,
	'ticker_start_end': ticker_start_end,
	'global_difficulty': GlobalDifficulty()
})
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
#{'batch_size': 128, 'learning_rate': 0.0003553661293493189, 'gamma': 0.9693339928617053, 'clip_range': 0.2151957180412349, 'n_epochs': 7}
ent_coef = 0.02  # Default: 0.0 (increase to encourage exploration)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps, tensorboard_log=log_dir, batch_size=batch_size, learning_rate=linear_schedule(2.5e-4), ent_coef=ent_coef, clip_range=clip_range, gamma=gamma, max_grad_norm=0.25, normalize_advantage=True)
model.learn(total_timesteps=300_000, progress_bar=True, tb_log_name="stock_ppo", callback=TensorboardCallback(), log_interval=1)
model.save("stock_ppo")
    