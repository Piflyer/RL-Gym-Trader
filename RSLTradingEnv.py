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

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

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
                use_privileged_obs=False,
                seed=0,
                max_episode_length=1000,
                device="cpu",
                rl_platform="SB3",
                debug=False,
                extra_obs=True,
                dataloader=None,
                eval_buffer = 0,
                min_percnt=0.8,
                eval = False,
                num_priv_obs=5,
                curriculum_manager=None,
                failed_trade_terminate = False,
                verbose=False):
        super().__init__()
        
        # ---- Initialize parameters ---- 
        
        self.cache_name = 'yfinance_cache'
        self.expire_after = datetime.timedelta(days=1) # Cache expires after 1 day

        # Create a cached session
        self.session = requests_cache.CachedSession(
            cache_name=self.cache_name,
            backend='sqlite',
            expire_after=self.expire_after
        )
        self.current_stock = "" # Current stock being traded
        self.initial_buy_date = "" # Initial buy date
        self.stock = stock
        self.initial_shares = initial_shares
        self.initial_buy_date = initial_buy_date
        self.min_hold_days = min_hold_days
        self.granularity = granularity
        self.period = period
        self.device = device
        self.seed = seed
        self.randomize_episode = randomize_episode
        self.use_privileged_obs = use_privileged_obs
        self.verbose = verbose
        self.failed_trade_terminate = failed_trade_terminate
        self.max_episode_length = max_episode_length
        self.rl_platform = rl_platform
        self.debug = debug
        self.eval_buffer = eval_buffer
        self.eval = eval
        self.extra_obs = extra_obs
        self.min_percnt = min_percnt
        self.num_priv_obs = num_priv_obs
        self.curriculum_manager = curriculum_manager
        
        
        # ---- Stock Data Placeholder ----
        self.stock_data = None
        self.vix_data = None
        self.gspc_data = 0
        self.merged_data = None
        self.all_stock_data = dataloader
        
        # --- Randomization Options ---
        self.random_symbols = [
            "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL", "META", "ADBE",
            "NFLX", "VOO", "FTEC", "TSLA", "JPM", "V", "UNH"
        ]
        
        # TODO: refactor fetch data to do front load and be more efficient with caching
        if self.randomize_episode:
            if self.all_stock_data is None:
                print("[INFO] Randomizing episode data for each env...")
                print("Getting VIX Data")
                vix_ticker = yf.Ticker("^VIX", session=self.session)
                self.vix_data = vix_ticker.history(period=period, auto_adjust=True)
                if self.vix_data.empty: print(f"[WARNING] Failed to fetch data for ^VIX.")

                print("Getting GSPC Data")
                gspc_ticker = yf.Ticker("^GSPC", session=self.session)
                self.gspc_data = gspc_ticker.history(period=period,  auto_adjust=True)
                if self.gspc_data.empty: print(f"[WARNING] Failed to fetch data for ^GSPC.")
                self.all_stock_data = {}
                for symbol in tqdm(self.random_symbols, desc="Fetching stock data"):
                    self.all_stock_data[symbol] = self.batch_fetch_data(symbol, self.period)
        if not self.randomize_episode:
            print("Getting VIX Data")
            vix_ticker = yf.Ticker("^VIX", session=self.session)
            self.vix_data = vix_ticker.history(period=period, auto_adjust=True)
            if self.vix_data.empty: print(f"[WARNING] Failed to fetch data for ^VIX.")
            print("Getting GSPC Data")
            gspc_ticker = yf.Ticker("^GSPC", session=self.session)
            self.gspc_data = gspc_ticker.history(period=period,  auto_adjust=True)
            if self.gspc_data.empty: print(f"[WARNING] Failed to fetch data for ^GSPC.")
        
        # --- Action Space ---
        self.action_space = gym.spaces.Discrete(2) # 0: Hold, 1: Buy/Sell
        
        # --- Observation Space ---
        obs_shape_base = 50 # Base observation shape
        if not self.extra_obs:
            obs_shape_base -= 14 # Remove extra observations
        if self.use_privileged_obs:
            obs_shape_base += self.num_priv_obs
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
        self.bought_price = 0.0 # Price at which the stock was bought
        self.cum_gain = 0.0 # Cumulative gain since the start of the episode
        self.holding_cum_gain = 0.0 # Cumulative gain since the start of the episode
        self.prev_close = 0.0 # Previous close price of the stock
        self.failed_buy = 0 # whether the buy failed
        self.action_counter = 0
        # ---- Fetch Data ----
        self._init_data()
        extras = {}
        extras["observations"] = {}
        # ---- Set initial state values AFTER data is loaded and index is set ----
        self.inital_bought_price = self.merged_data['Close'].iloc[self.ticker_index]
        self.bought_price = self.inital_bought_price # Set current bought price for the start
        self.current_close = self.inital_bought_price
        
        obs, priv = self.get_observations()
        extras["observations"]["actor"] = obs
        
        # ---- Continue setting up extras ----
        if self.use_privileged_obs:
            extras["observations"]["critic"] = priv["observations"]["critic"]
        extras["current_timestep"] = self.current_timestep
        extras["current_close"] = self.current_close
        extras["current_holding_value"] = self.current_holding_value
        extras["current_networth"] = self.current_networth
        extras["current_step"] = self.current_timestep   
        extras["has_position"] = self.has_position
        extras["reward"] = 0.0
        extras["holding_gains"] = (self.cum_gain - self.holding_cum_gain) / abs(self.holding_cum_gain) if self.holding_cum_gain != 0 else 0
        extras["net_gain"] = (self.current_networth - self.initial_networth) / abs(self.initial_networth) if self.initial_networth != 0 else 0
        end = time()
        if self.debug:
            print(f"Reset Time: {end - start:.4f}s")
        if self.rl_platform == "SB3":
            return obs.numpy(), extras
        else:
            return obs, extras
    
    def batch_fetch_data(self, stock, period):
        # print(f"[INFO] Fetching data for {stock}...")
        try:
            stock_ticker = yf.Ticker(stock, session=self.session)
            stock_data = stock_ticker.history(period=period, auto_adjust=True)
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
                print(f"[WARNING] Insufficient merged data after processing for {stock} (Final Length: {len(self.merged_data)}, Required: {min_required_length}).")
                return None
            
            return merged_data
        except Exception as e:
            print(f"[ERROR] Exception during data fetch/process for {stock}: {e}")
            return None
    
    def _fetch_data(self, stock, period):
        try:
            stock_ticker = yf.Ticker(stock, session=self.session)
            self.stock_data = stock_ticker.history(period=period, auto_adjust=True)
            if self.stock_data.empty: print(f"[WARNING] Failed to fetch data for primary stock {stock}.")

            if self.extra_obs:
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
            # self.merged_data = self.merged_data.dropna()
            
            self.merged_data.index = pd.to_datetime(self.merged_data.index) # Ensure index is datetime

            # --- More robust cleaning ---
            # Forward fill small gaps first (optional but can help)
            self.merged_data = self.merged_data.ffill(limit=3)
            # Handle NaNs in critical columns
            if 'Volume' in self.merged_data.columns:
                self.merged_data['Volume'].fillna(0.0, inplace=True) # Fill NaN volume with 0
            self.merged_data.dropna(subset=['Close'], inplace=True) # Drop rows ONLY if 'Close' is missing

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
        self.current_stock = np.random.choice(list(self.all_stock_data.keys()))
        self.merged_data = self.all_stock_data[self.current_stock]
        if self.merged_data is None or self.merged_data.empty:
            print(f"[ERROR] Failed to fetch data for {self.current_stock}.")
            self.reset()
        
        # if not self._fetch_data(self.current_stock, 'max'):
        #     print(f"[ERROR] Failed to fetch data for {self.current_stock}.")
        #     self.reset()
        self.ticker_index = np.random.randint(30, len(self.merged_data) - (self.eval_buffer + self.max_episode_length + 5))
        # Ensure eval starts at a fixed point with enough history and future data within the slice
        if self.eval:
            # Start after the initial 30 days needed for features (indices 0-29)
            # Ensure there's enough data for max_episode_length steps within the slice
            self.ticker_index = 30 # Start at the 31st data point (index 30)
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
            self.merged_data = self.batch_fetch_data(self.stock, self.period)
            if self.merged_data is None or self.merged_data.empty:
                print(f"[ERROR] Failed to fetch data for {self.stock}.")
                # end the episode
                # return
                raise RuntimeError(f"Failed to fetch data for {self.stock} during init.")
            # self.ticker_index = self.merged_data.index.get_loc(pd.to_datetime(self.initial_buy_date).date())
            try:
                requested_date = pd.to_datetime(self.initial_buy_date).normalize().date()
                # Find the first valid trading day index >= requested_date with enough history (30 days)
                min_required_index_loc = 30
                valid_indices = np.where(self.merged_data.index.date >= requested_date)[0]
                suitable_indices = valid_indices[valid_indices >= min_required_index_loc]

                if len(suitable_indices) > 0:
                    self.ticker_index = suitable_indices[0]
                    self.initial_buy_date = self.merged_data.index[self.ticker_index].date() # Update to actual date used
                    if self.debug: print(f"[DEBUG] Using start date {self.initial_buy_date} for fixed episode.")
                else:
                    raise ValueError(f"No suitable start date found >= {requested_date} with {min_required_index_loc} days history.")
            except Exception as e:
                raise ValueError(f"Error setting initial buy date for '{self.initial_buy_date}': {e}")
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
        if len(price_array) == 5:
            denom_5 = price_array[-5]
            if not pd.isna(denom_5) and abs(denom_5) > 1e-9: # Check for NaN and near-zero
                momentum_5 = (price_array[-1] - denom_5) / denom_5
            else:
                momentum_5 = 0.0
            return momentum_5 # for privileged observation ONLY
        
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
        - Gains over holding (1)
        - Current position (1)
        Total: 50 features
        
        Privileged observation (if use_privileged_obs is True):
        - Next 5 closing prices of the stock (5)
        - Next 5 closing prices of the VIX (5)
        - Next 5 closing prices of the S&P 500 (5)
        - Momentum of the stock (1)
        - Momentum of the VIX (1)
        - Momentum of the S&P500 (1)
        
        Total: 50 + 18 = 68 features
        returns: torch.Tensor, dict()
        """
        
        stock_30 = self._fetch_closing("stock", 30)
        stock_7_momentum, stock_30_momentum = self._calculate_momentum(stock_30)
        if self.extra_obs:
            vix_5 = self._fetch_closing("vix", 5)
            gspc_5 = self._fetch_closing("gspc", 5)
            vix_7_momentum, vix_30_momentum = self._calculate_momentum(self._fetch_closing("vix", 30))
            gspc_7_momentum, gspc_30_momentum = self._calculate_momentum(self._fetch_closing("gspc", 30))
        current_relative_gain = (self.merged_data['Close'].iloc[self.ticker_index] - self.bought_price) / abs(self.bought_price)
        if not pd.isna(self.bought_price) and abs(self.bought_price) > 1e-9:
            current_relative_gain = (self.merged_data['Close'].iloc[self.ticker_index] - self.bought_price) / abs(self.bought_price) 
        if not self.has_position:
            current_relative_gain *= -1
        
        if self.ticker_index - 1 < 0:
            volume_change = 0.0
        else:
            prev_volume = self.merged_data['Volume'].iloc[self.ticker_index - 1]
        if prev_volume == 0:
            volume_change = 0.0
        if not pd.isna(prev_volume) and abs(prev_volume) > 1e-9:
            volume_change = (self.merged_data['Volume'].iloc[self.ticker_index] - prev_volume) / prev_volume
        volume_change = np.nan_to_num(volume_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        #more observations for robustness?
        
        gains_over_holding = (self.cum_gain - self.holding_cum_gain) / abs(self.holding_cum_gain) if self.holding_cum_gain != 0 else 0
        position = 1 if self.has_position else 0
        # Normalize the stock_30 prices
        stock_30 = (stock_30 - np.mean(stock_30)) / np.std(stock_30)
        # Normalize the vix_5 prices
        vix_5 = (vix_5 - np.mean(vix_5)) / np.std(vix_5)
        # Normalize the gspc_5 prices
        gspc_5 = (gspc_5 - np.mean(gspc_5)) / np.std(gspc_5)
        
        #current holding value
        current_holding_value = 0
        if self.has_position:
            current_holding_value = self.current_holding_value
        # Normalize the current holding value
        current_holding_value = (current_holding_value - np.mean(stock_30)) / np.std(stock_30)
        
        #current networth
        current_networth = (self.current_networth - np.mean(stock_30)) / np.std(stock_30)
        
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
                [volume_change],
                [gains_over_holding],
                [position],
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
                priv_vix_5_momentum = self._calculate_momentum(priv_vix)
                priv_gspc_5_momentum = self._calculate_momentum(priv_gspc)
                priv_stock_5_momentum = self._calculate_momentum(priv_stock)
                priv_vix = (priv_vix - np.mean(vix_5)) / np.std(vix_5)
                priv_gspc = (priv_gspc - np.mean(gspc_5)) / np.std(gspc_5)
                priv_stock = (priv_stock - np.mean(stock_30)) / np.std(stock_30)
                privileged_obs = np.concatenate([base_obs, priv_stock, priv_vix, priv_gspc, [priv_stock_5_momentum], [priv_vix_5_momentum], [priv_gspc_5_momentum]])
            else:
                privileged_obs = np.concatenate([base_obs, priv_stock])
            privileged_obs = np.reshape(privileged_obs, (-1, 1))
        
        base_obs = np.reshape(base_obs, (-1, 1))
        base_obs = np.nan_to_num(base_obs, nan=0.0, posinf=0.0, neginf=0.0) 
        base_obs = torch.tensor(base_obs, dtype=torch.float32, device=self.device)

        extras = {}
        extras["observations"] = {}
        extras["observations"]["actor"] = base_obs
        if self.use_privileged_obs:
            privileged_obs = np.reshape(privileged_obs, (-1, 1))
            privileged_obs = np.nan_to_num(privileged_obs, nan=0.0, posinf=0.0, neginf=0.0)
            privileged_obs = torch.tensor(privileged_obs, dtype=torch.float32, device=self.device)
            extras['observations']['critic'] = privileged_obs
        else:
            extras['observations']['critic'] = base_obs
        
        if self.use_privileged_obs and self.rl_platform == "SB3":
            return privileged_obs, extras
        #check if nan is in the base_obs
        if torch.isnan(base_obs).any():
            print(f"[ERROR] NaN detected in base_obs: {base_obs}")
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
        self.failed_buy = 0
        
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
                    if self.failed_trade_terminate:
                        terminated = True
                    # terminated = True
                    #means that trade doesn't go through [OVERRIDE]
                    self.sold_price = 0
                    self.failed_buy = 1
                    self.days_since_last_trade += 1
                    action = 0
                else:
                    # self.current_holding_value = 0
                    self.sold_price = self.current_close
                    self.days_since_last_trade = 0
            if self.failed_buy == 0:
                self.action_counter += 1
        
        if self.current_networth < self.initial_networth * self.min_percnt:
            terminated = True
        
        # Check if the episode is done
        if self.current_timestep >= self.max_episode_length:
            truncated = True
        
        # Calculate the reward
        self.prev_close = self.merged_data['Close'].iloc[self.ticker_index - 1]
        self.holding_cum_gain  += (self.current_close - self.prev_close) * self.initial_shares
        if self.has_position:
            self.cum_gain += (self.current_close - self.prev_close) * self.initial_shares
        reward = self._reward(action)
        #order MATTERS for cum_gain and max_profit_since_buy
        if self.has_position:
            self.max_profit_since_buy = max(self.max_profit_since_buy, self.current_holding_value)
        else:
            self.max_profit_since_buy = min(self.max_profit_since_buy, -self.current_holding_value)
        if action == 1:
            self.has_position = not self.has_position
            if self.has_position: # if we just bought
                self.max_profit_since_buy = 0 
            else: # if we just sold
                self.max_profit_since_buy = self.current_holding_value
            self.current_holding_value = 0 # cleanup after buy/sell

        
        # Get the next observation
        self.merged_data.fillna(0, inplace=True)
        observation, extras = self.get_observations()
        
        # Update the current timestep
        self.current_timestep += 1
        self.ticker_index += 1
        dones = [terminated, truncated]
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        extras["current_timestep"] = self.current_timestep
        extras["current_close"] = self.current_close
        extras["current_holding_value"] = self.current_holding_value
        extras["current_networth"] = self.current_networth
        extras["current_step"] = self.current_timestep
        extras["has_position"] = self.has_position
        extras["holding_gains"] = (self.cum_gain - self.holding_cum_gain) / abs(self.holding_cum_gain) if self.holding_cum_gain != 0 else 0
        extras["net_gain"] = (self.current_networth - self.initial_networth) / abs(self.initial_networth) if self.initial_networth != 0 else 0
        extras["reward"] = reward
        end = time()
        if self.debug:
            print(f"Step Time: {end - start:.4f}s")
        if self.rl_platform == "SB3":
            return observation.numpy(), reward, terminated, truncated, extras
        else:
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
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
            denominator = abs(self.max_profit_since_buy) + 1e-9
            # self.greedy_reward = (self.current_holding_value - self.max_profit_since_buy) / denominator
            self.greedy_reward = (self.current_holding_value - self.max_profit_since_buy) / self.current_networth
        if self.max_profit_since_buy != 0 and not self.has_position:
            denominator = abs(self.max_profit_since_buy) + 1e-9
            # self.greedy_reward = (-self.current_holding_value + self.max_profit_since_buy) / denominator
            self.greedy_reward = (self.max_profit_since_buy - self.current_holding_value) / self.current_networth
        if self.max_profit_since_buy == 0:
            self.greedy_reward = 0.0
        
        
        
        # Calculate the the current networth gain to if just held the stock
        if self.holding_cum_gain != 0:   
            if self.min_hold_days > 0:
                self.holding_reward = (self.cum_gain - self.holding_cum_gain) / abs(self.holding_cum_gain)
        else:
            self.holding_reward = 0.1
        
        #werighted greedy reward by percentage of the current portfolio
        self.sign = 1
        if self.greedy_reward < 0:
            self.sign = -1
        
        #penalty for selling too early:
        self.min_hold_penalty = 0
        if (self.days_since_last_trade < self.min_hold_days) and action == 1:
            self.min_hold_penalty = (self.days_since_last_trade - self.min_hold_days) / self.min_hold_days
        
        # consturct the reward
        self.weighted_greedy = self.sign * ((abs(self.greedy_reward) + 1)**2 -1)
        self.clipped_weighted_greedy = np.clip(self.weighted_greedy, -1, 1)
        self.clipped_holding_reward = np.clip(self.holding_reward, -1, 1)
        
        if self.curriculum_manager is not None:
            curriculum_weight = self.curriculum_manager.get_curriculum("greedy_reward") 
            weight = curriculum_weight / self.curriculum_manager.get_max_steps()
            self.clipped_weighted_greedy = self.clipped_weighted_greedy * weight
        
        self.failed_buy_reward = self.failed_buy * -2
        
        reward = self.clipped_weighted_greedy + self.clipped_holding_reward + self.min_hold_penalty +  self.failed_buy_reward
        
        #clip the reward
        reward = np.clip(reward, -3, 3)
        
        return reward

if __name__ == "__main__":
    env = TradingEnv(randomize_episode=False, initial_buy_date="2020-12-03", stock="UAL", extra_obs=True)
    obs, extras = env.reset()
    cum_reward = 0
    for j in range(100):
        action = env.action_space.sample()
        obs, reward, truncated, terminated, info = env.step(action)
        print(" ")
        print(f"Current Holding Value: {env.current_holding_value}, Current Networth: {env.current_networth}")
        print(f"Greedy Reward: {env.greedy_reward}, Holding Reward: {env.holding_reward}")
        print(f"Weighted Greedy Reward: {env.weighted_greedy}, Min Hold Penalty: {env.min_hold_penalty}")
        print(f"Cumulative Gain: {env.cum_gain}, Holding Cumulative Gain: {env.holding_cum_gain}")
        print(f"Reward: {reward}, Action: {action}, Postion: {env.has_position}")
        print(" ")
        cum_reward += reward
    
    print(f"Cumulative Reward: {cum_reward}")
