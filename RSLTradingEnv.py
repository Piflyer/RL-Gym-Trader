import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from time import time

import requests_cache
import datetime

cache_name = 'yfinance_cache'
expire_after = datetime.timedelta(days=1) # Cache expires after 1 day

# Create a cached session
session = requests_cache.CachedSession(
    cache_name=cache_name,
    backend='sqlite',
    expire_after=expire_after
)

# Suppress specific pandas warnings if necessary (optional)
# warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.indexing")
# Ignore SettingWithCopyWarning if it becomes noisy during development
pd.options.mode.chained_assignment = None # default='warn'

class TradingEnv(gym.Env):
    """
    A Gymnasium environment for simulating stock trading.
    (Version 3.1: Cleaned for training, timezone fix, robust fetch)

    Args:
        stock (str): The ticker symbol of the main stock to trade. Default: 'AAPL'.
        initial_shares (int): The number of shares to start with. Default: 3.
        initial_buy_date (str): The date ('YYYY-MM-DD') to simulate the initial purchase.
                                Must be a valid trading day within the fetched data. Default: "2023-12-01".
        min_hold_days (int): Minimum number of days to hold before selling without penalty. Default: 10.
        granularity (str): Data granularity ('1d', '1h', '30m', etc.). Default: '1d'.
        period (str): Data period ('max', '1y', '6mo', etc.). Default: 'max'.
        randomize_episode (bool): Whether to randomize stock, start date, and shares at the beginning of each episode. Default: True.
        use_privileged_obs (bool): If True, includes the next 5 days' closing prices in the observation (intended for critic). Default: False.
        seed (int): Random seed for reproducibility. Default: 0.
        verbose (bool): If True, prints step action/reward information. Default: False.
    """
    def __init__(self,
                stock='AAPL',
                initial_shares=3,
                initial_buy_date="2023-12-03",
                min_hold_days=10,
                granularity='1d',
                period='max',
                randomize_episode=True,
                use_privileged_obs=True,
                seed=0,
                max_episode_length=1000,
                device="cpu",
                rl_platform="SB3",
                debug=False,
                extra_obs=False,
                verbose=False):
        super().__init__()
        
        # ---- Initialize parameters ---- 
        self.current_stock = "" # Current stock being traded
        self.initial_buy_date = "" # Initial buy date
        self.stock = stock
        self.initial_shares = initial_shares
        self.initial_buy_date = initial_buy_date
        self.min_hold_days = min_hold_days
        self.granularity = granularity
        self.period = period
        self.device = device
        self.randomize_episode = randomize_episode
        self.use_privileged_obs = use_privileged_obs
        self.verbose = verbose
        self.max_episode_length = max_episode_length
        self.rl_platform = rl_platform
        self.debug = debug
        self.extra_obs = extra_obs
        
        
        # ---- Stock Data Placeholder ----
        self.stock_data = None
        self.vix_data = None
        self.gspc_data = 0
        
        # --- Randomization Options ---
        self.random_symbols = [
            "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL", "META", "ADBE",
            "NFLX", "VOO", "FTEC", "TSLA", "JPM", "V", "UNH"
        ]
        
        # --- Action Space ---
        self.action_space = gym.spaces.Discrete(2) # 0: Hold, 1: Buy/Sell
        
        # --- Observation Space ---
        obs_shape_base = 48 # Base observation shape
        if not self.extra_obs:
            obs_shape_base -= 14 # Remove extra observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape_base, 1), dtype=np.float32
        )
        self.reset(seed=None, options=None)
        
        # for RSL_RL compatibility
        self.num_actions = self.action_space.n
        self.num_observations = self.observation_space.shape[0]
        self.num_envs = 32 # Number of environments for RSL_RL
        self.cfg = {
            "env": {
                "num_envs": self.num_envs,
                "num_actions": self.num_actions,
                "num_observations": self.num_observations,
                "action_space": self.action_space,
                "observation_space": self.observation_space,
                "max_episode_length": self.max_episode_length,
                "is_privileged": self.use_privileged_obs,
            }
        }
        
        
    
    def reset(self, seed=None, options=None):
        start = time()
        # ---- Action Space ----
        self.current_timestep = 0 # Current timestep
        self.initial_networth = 0.0 # Initial net worth
        self.current_networth = 0.0 # Current net worth cumulated
        self.bought_price = 0.0 # Price at which the stock was bought
        self.current_holding_value = 0.0 # Current value of the holding
        self.days_since_last_trade = 0 # Days since the last trade
        self.has_position = True # Whether the agent currently holds a position
        self.max_profit_since_buy = 0.0 # Maximum profit since the stock was bought  
        self.ticker_index = 0 # Index of the current data index in the data
        self.current_close = 0.0 # Current close price of the stock
        self.sold_price = 0.0 # Price at which the stock was sold
        self.inital_bought_price = 0.0 # Price at which the stock was bought
        self.cum_gain = 0.0 # Cumulative gain since the start of the episode
        # ---- Fetch Data ----
        self._init_data()
        obs, priv = self.get_observations()
        extras = {}
        extras["observations"] = {}
        extras["observations"]["actor"] = obs
        if self.use_privileged_obs:
            extras["observations"]["critic"] = priv["observations"]["critic"]
        extras["current_timestep"] = self.current_timestep
        extras["current_close"] = self.current_close
        extras["current_holding_value"] = self.current_holding_value
        extras["current_networth"] = self.current_networth
        extras["current_step"] = self.current_timestep   
        extras["has_position"] = self.has_position
        self.inital_bought_price = self.merged_data['Close'].iloc[self.ticker_index]
        self.current_close = self.merged_data['Close'].iloc[self.ticker_index]
        end = time()
        if self.debug:
            print(f"Reset Time: {end - start:.4f}s")
        if self.rl_platform == "SB3":
            return obs.numpy(), extras
        else:
            return obs, extras
        
    def _fetch_data(self, stock, period):
        try:
            stock_ticker = yf.Ticker(stock, session=session)
            self.stock_data = stock_ticker.history(period=period, auto_adjust=True)
            if self.stock_data.empty: print(f"[WARNING] Failed to fetch data for primary stock {stock}.")

            if self.extra_obs:
                vix_ticker = yf.Ticker("^VIX", session=session)
                self.vix_data = vix_ticker.history(period=period, auto_adjust=True)
                if self.vix_data.empty: print(f"[WARNING] Failed to fetch data for ^VIX.")

                gspc_ticker = yf.Ticker("^GSPC", session=session)
                self.gspc_data = gspc_ticker.history(period=period,  auto_adjust=True)
                if self.gspc_data.empty: print(f"[WARNING] Failed to fetch data for ^GSPC.")

                stock_df = self.stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not self.stock_data.empty else pd.DataFrame(index=self.stock_data.index)
                vix_df = self.vix_data[['Close']].rename(columns={'Close': 'VIX_Close'}).copy() if not self.vix_data.empty else pd.DataFrame(index=self.vix_data.index)
                gspc_df = self.gspc_data[['Close']].rename(columns={'Close': 'GSPC_Close'}).copy() if not self.gspc_data.empty else pd.DataFrame(index=self.gspc_data.index)

                self.merged_data = pd.concat([stock_df, vix_df, gspc_df], axis=1, join='outer')
                if 'Close' not in self.merged_data.columns and not stock_df.empty:
                    print("[ERROR] Primary stock 'Close' column missing after outer join.")
            else:
                stock_df = self.stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not self.stock_data.empty else pd.DataFrame(index=self.stock_data.index)
                self.merged_data = stock_df

            self.merged_data = self.merged_data.groupby(self.merged_data.index.date).first()
            self.merged_data = self.merged_data.dropna()

            min_required_length = 35 + self.max_episode_length
            if len(self.merged_data) < min_required_length:
                print(f"[WARNING] Insufficient merged data after processing for {stock} (Final Length: {len(self.merged_data)}, Required: {min_required_length}).")
                return False
            
            return True
        except Exception as e:
            print(f"[ERROR] Exception during data fetch/process for {stock}: {e}")
            self.merged_data = pd.DataFrame()
            return False
        
    def _randomize_data(self):
        """
        Randomizes the stock data and initial buy date for each episode.
        """
        self.current_stock = np.random.choice(self.random_symbols)
        if not self._fetch_data(self.current_stock, 'max'):
            print(f"[ERROR] Failed to fetch data for {self.current_stock}.")
            self.reset()
        self.ticker_index = np.random.randint(30, len(self.merged_data) - self.max_episode_length - 5)
        self.initial_buy_date = self.merged_data.index[self.ticker_index]
        self.initial_shares = np.random.randint(1, 11)
        self.initial_networth = self.initial_shares * self.merged_data['Close'].iloc[self.ticker_index]
        self.current_networth = self.initial_networth
        self.current_holding_value = self.initial_networth
        self.bought_price = self.merged_data['Close'].iloc[self.ticker_index]
    
    def _init_data(self):
        if self.randomize_episode:
            self._randomize_data()
        else:
            if not self._fetch_data(self.stock, self.period):
                print(f"[ERROR] Failed to fetch data for {self.stock}.")
                # end the episode
                return
            self.ticker_index = self.merged_data.index.get_loc(pd.to_datetime(self.initial_buy_date).date())
            self.initial_networth = self.initial_shares * self.merged_data['Close'].iloc[self.ticker_index]
            self.current_networth = self.initial_networth
            self.current_holding_value = self.initial_networth
            self.bought_price = self.merged_data['Close'].iloc[self.ticker_index]
    
    def _fetch_closing(self, stock, period):
        """Fetches the closing prices for a given stock and period.
        Args:
            stock (str): The ticker symbol of the stock.
            period (str): The period for which to fetch the data.
        Returns:
            np.ndarray: An array of closing prices
        """
        
        # get the closing prices for the stock from self.ticker_index - period 
        if stock == "stock":
            return self.merged_data['Close'].iloc[self.ticker_index - period:self.ticker_index].values
        if stock == "vix":
            return self.merged_data['VIX_Close'].iloc[self.ticker_index - period:self.ticker_index].values
        if stock == "gspc":
            return self.merged_data['GSPC_Close'].iloc[self.ticker_index - period:self.ticker_index].values
        else:
            raise ValueError(f"Invalid stock name: {stock}. Must be 'stock', 'vix', or 'gspc'.")
    
    def _calculate_momentum(self, price_array):
        if len(price_array) < 30:
            # Return 0.0 or handle as appropriate if not enough data
            return 0.0, 0.0 # Or raise an error earlier

        momentum_7 = 0.0
        denom_7 = price_array[-7]
        if not pd.isna(denom_7) and abs(denom_7) > 1e-9: # Check for NaN and near-zero
            momentum_7 = (price_array[-1] - denom_7) / denom_7

        momentum_30 = 0.0
        denom_30 = price_array[-30]
        if not pd.isna(denom_30) and abs(denom_30) > 1e-9: # Check for NaN and near-zero
            momentum_30 = (price_array[-1] - denom_30) / denom_30

        # Replace potential inf/-inf with 0 (or a large capped value)
        momentum_7 = np.nan_to_num(momentum_7, nan=0.0, posinf=0.0, neginf=0.0)
        momentum_30 = np.nan_to_num(momentum_30, nan=0.0, posinf=0.0, neginf=0.0)

        return momentum_7, momentum_30

    
    def get_observations(self):
        """Get the current observation. 
        The observation includes:
        - Past 30 closing prices of the stock (30)
        - Past 5 closing prices of the VIX (5)
        - Past 5 closing prices of the S&P 500 (5)
        - Past 7 day momentum of the stock (1)
        - Past 30 day momentum of the stock (1)
        - Past 7 day momentum of the VIX (1)
        - Past 30 day momentum of the VIX (1)
        - Past 7 day momentum of the S&P500 (1)
        - Current relative gain (percentage) (1)
        - Percentage change of volume (1)
        Total: 47 features
        
        Privileged observation (if use_privileged_obs is True):
        - Next 5 closing prices of the stock (5)
        - Next 5 closing prices of the VIX (5)
        - Next 5 closing prices of the S&P 500 (5)
        
        Total: 47 + 15 = 62 features
        returns: torch.Tensor, dict()
        """
        
        stock_30 = self._fetch_closing("stock", 30)
        stock_7_momentum, stock_30_momentum = self._calculate_momentum(stock_30)
        if self.extra_obs:
            vix_5 = self._fetch_closing("vix", 5)
            gspc_5 = self._fetch_closing("gspc", 5)
            vix_7_momentum, vix_30_momentum = self._calculate_momentum(self._fetch_closing("vix", 30))
            gspc_7_momentum, gspc_30_momentum = self._calculate_momentum(self._fetch_closing("gspc", 30))
        current_relative_gain = (self.merged_data['Close'].iloc[self.ticker_index] - self.bought_price) / self.bought_price
        current_relative_gain = np.nan_to_num(current_relative_gain, nan=0.0, posinf=0.0, neginf=0.0)
        if self.merged_data['Volume'].iloc[self.ticker_index - 1] == 0:
            volume_change = 0
        else:
            volume_change = (self.merged_data['Volume'].iloc[self.ticker_index] - self.merged_data['Volume'].iloc[self.ticker_index - 1]) / self.merged_data['Volume'].iloc[self.ticker_index - 1]
        
        # Privileged observation
        if self.extra_obs:
            base_obs = np.concatenate([
                stock_30,
                vix_5,
                gspc_5,
                [stock_7_momentum, stock_30_momentum],
                [vix_7_momentum, vix_30_momentum],
                [gspc_7_momentum, gspc_30_momentum],
                [current_relative_gain],
                [volume_change]
            ])
        else:
            base_obs = np.concatenate([
                stock_30,
                [stock_7_momentum, stock_30_momentum],
                [current_relative_gain],
                [volume_change]
            ])
        
        if self.use_privileged_obs:
            priv_stock = self._fetch_closing("stock", 5)
            if self.extra_obs:
                priv_vix = self._fetch_closing("vix", 5)
                priv_gspc = self._fetch_closing("gspc", 5)
                privileged_obs = np.concatenate([base_obs, priv_stock, priv_vix, priv_gspc])
            else:
                privileged_obs = np.concatenate([base_obs, priv_stock])
            privileged_obs = np.reshape(privileged_obs, (-1, 1))
        
        base_obs = np.reshape(base_obs, (-1, 1))
        base_obs = np.nan_to_num(base_obs, nan=0.0, posinf=0.0, neginf=0.0) 
        base_obs = torch.tensor(base_obs, dtype=torch.float32)
        extras = {}
        extras["observations"] = {}
        extras["observations"]["actor"] = base_obs
        if self.use_privileged_obs:
            privileged_obs = np.reshape(privileged_obs, (-1, 1))
            privileged_obs = np.nan_to_num(privileged_obs, nan=0.0, posinf=0.0, neginf=0.0) 
            privileged_obs = torch.tensor(privileged_obs, dtype=torch.float32)
            extras['observations']['critic'] = privileged_obs
        else:
            extras['observations']['critic'] = base_obs
        
        return base_obs, extras
        
    def step(self, action):
        start = time()
        """Takes in an action and returns the next observation, reward, done, and info.
        0: Hold, 1: Buy/Sell
        
        Args:
            action (torch.tensor): The action to take.
        Output:
            observation (torch.tensor): The next observation.
            reward (float): The reward for the action taken.
            done (bool): Whether the episode is done.
            info (dict): Additional information about the step.
        """
        if type(action) == torch.Tensor:
            action = action.item()
        self.current_close = self.merged_data['Close'].iloc[self.ticker_index]
        self.current_holding_value = (self.current_close - self.bought_price) * self.initial_shares
        terminated = False
        truncated = False
        if self.current_timestep == 0:
            self.inital_bought_price = self.current_close

        if action == 0:
            self.sold_price = 0
            self.days_since_last_trade += 1
    
        else:
            self.days_since_last_trade = 0
            if self.has_position:
                # if self.days_since_last_trade < self.min_hold_days:
                #     terminated = True
                self.current_networth += self.current_holding_value
                self.sold_price =  self.current_close
                # self.current_holding_value = 0
                self.days_since_last_trade = 0
                # we are keeping holding value to help calculate the reward
            else:
                self.bought_price = self.current_close
                if self.current_networth < self.current_close * self.initial_shares:
                    terminated = True
                # self.current_holding_value = 0
                self.sold_price = self.current_close
                self.days_since_last_trade = 0
        
        if self.current_networth < self.initial_networth * 0.9:
            terminated = True
        
        # Check if the episode is done
        if self.current_timestep >= self.max_episode_length:
            truncated = True
        
        # Calculate the reward
        if self.has_position:
            self.max_profit_since_buy = max(self.max_profit_since_buy, self.current_holding_value)
            prev_close = self.merged_data['Close'].iloc[self.ticker_index - 1]
            self.cum_gain += (self.current_close - prev_close) * self.initial_shares
        else:
            self.max_profit_since_buy = max(self.max_profit_since_buy, -self.current_holding_value)
        reward = self._reward(action)
        if action == 1:
            self.has_position = not self.has_position
            self.current_holding_value = 0 # cleanup after buy/sell
            self.max_profit_since_buy = 0 # cleanup after buy/sell
        
        # Get the next observation
        observation, extras = self.get_observations()
        
        # Update the current timestep
        self.current_timestep += 1
        self.ticker_index += 1
        dones = [terminated, truncated]
        dones = torch.tensor(dones, dtype=torch.bool)
        extras["current_timestep"] = self.current_timestep
        extras["current_close"] = self.current_close
        extras["current_holding_value"] = self.current_holding_value
        extras["current_networth"] = self.current_networth
        extras["current_step"] = self.current_timestep
        extras["has_position"] = self.has_position
        end = time()
        if self.debug:
            print(f"Step Time: {end - start:.4f}s")
        if self.rl_platform == "SB3":
            return observation.numpy(), reward, terminated, truncated, extras
        else:
            reward = torch.tensor(reward, dtype=torch.float32)
            return observation, reward, dones, extras
    
    def _reward(self, action):
        """Calculates the reward based on the current state.
        The reward is calculated as the difference between the current net worth and the initial net worth.
        """
        
        #Greedy Reward: trying to maximize the current net worth through each step
        # if has_position try to maximize the current holding value and penalize if current holding is below max
        # if doesnt't have position, try to maximize loss 
        self.greedy_reward = 0
        if self.max_profit_since_buy != 0 and self.has_position:
            self.greedy_reward = (self.current_holding_value - self.max_profit_since_buy)/ self.max_profit_since_buy
        if self.max_profit_since_buy != 0 and not self.has_position:
            self.greedy_reward = (-self.current_holding_value + self.max_profit_since_buy)/ self.max_profit_since_buy
        if self.max_profit_since_buy == 0:
            self.greedy_reward = 0.3
        
        
        
        # Calculate the the current networth gain to if just held the stock
        self.holding = (self.current_close - self.inital_bought_price) * self.initial_shares
        if self.has_position and self.holding != 0:   
            self.holding_reward = (self.cum_gain - self.holding) / abs(self.holding)
        else:
            self.holding_reward = 0.1
        
        #penalty for selling too early:
        self.min_hold_penalty = 0
        if (self.days_since_last_trade < self.min_hold_days) and action == 1:
            self.min_hold_penalty = (self.days_since_last_trade - self.min_hold_days) / self.min_hold_days
        
        # consturct the reward
        
        
        reward = 2.0* self.greedy_reward + 3.0*self.holding_reward + 2.0 * self.min_hold_penalty
        
        #clip the reward
        reward = np.clip(reward, -10, 7)
        
        return reward  
        
        
        
        
if __name__ == "__main__":
    env = TradingEnv(debug=True)
    obs, extras = env.reset()
    for j in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, extras = env.step(action)
        if done or truncated:
            obs, extras = env.reset()
            
        
            
            
        
        
        
        
        
        
        
        