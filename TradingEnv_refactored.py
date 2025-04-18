import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

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
        self.randomize_episode = randomize_episode
        self.use_privileged_obs = use_privileged_obs
        self.verbose = verbose
        self.max_episode_length = max_episode_length
        
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
        obs_shape_base = 47 # Base observation shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape_base, 1), dtype=np.float32
        )
        self.reset(seed=None, options=None)
        
    
    def reset(self, seed=None, options=None):
        # ---- Action Space ----
        self.current_timestep = 0 # Current timestep
        self.initial_networth = 0.0 # Initial net worth
        self.current_networth = 0.0 # Current net worth cumulated
        self.bought_price = 0.0 # Price at which the stock was bought
        self.current_holding_value = 0.0 # Current value of the holding
        self.days_since_last_trade = 0 # Days since the last trade
        self.has_position = False # Whether the agent currently holds a position
        self.max_profit_since_buy = 0.0 # Maximum profit since the stock was bought  
        self.ticker_index = 0 # Index of the current data index in the data
        self.current_close = 0.0 # Current close price of the stock
        self.sold_price = 0.0 # Price at which the stock was sold
        self.inital_bought_price = 0.0 # Price at which the stock was bought
        # ---- Fetch Data ----
        self._init_data()
        
    def _fetch_data(self, stock, period):
        try:
            stock_ticker = yf.Ticker(stock)
            self.stock_data = stock_ticker.history(period=period, auto_adjust=True)
            if self.stock_data.empty: print(f"[WARNING] Failed to fetch data for primary stock {stock}.")

            vix_ticker = yf.Ticker("^VIX")
            self.vix_data = vix_ticker.history(period=period, auto_adjust=True)
            if self.vix_data.empty: print(f"[WARNING] Failed to fetch data for ^VIX.")

            gspc_ticker = yf.Ticker("^GSPC")
            self.gspc_data = gspc_ticker.history(period=period,  auto_adjust=True)
            if self.gspc_data.empty: print(f"[WARNING] Failed to fetch data for ^GSPC.")

            stock_df = self.stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not self.stock_data.empty else pd.DataFrame(index=self.stock_data.index)
            vix_df = self.vix_data[['Close']].rename(columns={'Close': 'VIX_Close'}).copy() if not self.vix_data.empty else pd.DataFrame(index=self.vix_data.index)
            gspc_df = self.gspc_data[['Close']].rename(columns={'Close': 'GSPC_Close'}).copy() if not self.gspc_data.empty else pd.DataFrame(index=self.gspc_data.index)

            self.merged_data = pd.concat([stock_df, vix_df, gspc_df], axis=1, join='outer')
            if 'Close' not in self.merged_data.columns and not stock_df.empty:
                print("[ERROR] Primary stock 'Close' column missing after outer join.")

            self.merged_data = self.merged_data.groupby(self.merged_data.index.date).first()
            self.merged_data = self.merged_data.dropna()

            min_required_length = 35 + self.max_episode_length
            if len(self.merged_data) < min_required_length:
                print(f"[WARNING] Insufficient merged data after processing for {stock} (Final Length: {len(self.merged_data)}, Required: {min_required_length}).")

        except Exception as e:
            print(f"[ERROR] Exception during data fetch/process for {stock}: {e}")
            self.merged_data = pd.DataFrame()
        
    def _randomize_data(self):
        """
        Randomizes the stock data and initial buy date for each episode.
        """
        self.current_stock = np.random.choice(self.random_symbols)
        self._fetch_data(self.current_stock, 'max')
        self.ticker_index = np.random.randint(0, len(self.merged_data) - self.max_episode_length)
        self.initial_buy_date = self.merged_data.index[self.ticker_index]
        self.initial_shares = np.random.randint(1, 11)
        self.initial_networth = self.initial_shares * self.merged_data['Close'].iloc[self.ticker_index]
        self.current_networth = self.initial_networth
        self.current_holding_value = self.initial_networth
        self.bought_price = self.merged_data['Close'].iloc[self.ticker_index]
    
    def _init_data(self):
        self._fetch_data(self.stock, self.period)
        if self.randomize_episode:
            self._randomize_data()
        else:
            self._fetch_data(self.stock, self.period)
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
            return self.merged_data['Close'].iloc[self.ticker_index:self.ticker_index + period].values
        if stock == "vix":
            return self.merged_data['VIX_Close'].iloc[self.ticker_index:self.ticker_index + period].values
        if stock == "gspc":
            return self.merged_data['GSPC_Close'].iloc[self.ticker_index:self.ticker_index + period].values
        else:
            raise ValueError(f"Invalid stock name: {stock}. Must be 'stock', 'vix', or 'gspc'.")
    
    def _calculate_momentum(self, price_array):
        """ Calculates the momentum of a given price array.
        Args:
            price_array (np.ndarray): An array of closing prices.
        Returns:
            tuple: A tuple containing the 7-day and 30-day momentum values.
        """
        if len(price_array) < 30:
            raise ValueError("Price array must have at least 30 elements.")
        
        momentum_7 = (price_array[-1] - price_array[-8]) / price_array[-8]
        momentum_30 = (price_array[-1] - price_array[-31]) / price_array[-31]
        
        return momentum_7, momentum_30
    
    def get_observation(self):
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
        vix_5 = self._fetch_closing("vix", 5)
        gspc_5 = self._fetch_closing("gspc", 5)
        stock_7_momentum, stock_30_momentum = self._calculate_momentum(stock_30)
        vix_7_momentum, vix_30_momentum = self._calculate_momentum(vix_5)
        gspc_7_momentum, gspc_30_momentum = self._calculate_momentum(gspc_5)
        current_relative_gain = (self.merged_data['Close'].iloc[self.ticker_index] - self.bought_price) / self.bought_price
        volume_change = (self.merged_data['Volume'].iloc[self.ticker_index] - self.merged_data['Volume'].iloc[self.ticker_index - 1]) / self.merged_data['Volume'].iloc[self.ticker_index - 1]
        
        # Privileged observation
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
        
        if self.use_privileged_obs:
            priv_stock = self._fetch_closing("stock", 5)
            priv_vix = self._fetch_closing("vix", 5)
            priv_gspc = self._fetch_closing("gspc", 5)
            privileged_obs = np.concatenate(base_obs, [priv_stock, priv_vix, priv_gspc])
            privileged_obs = np.reshape(privileged_obs, (-1, 1))
        
        base_obs = np.reshape(base_obs, (-1, 1))
        
        base_obs = torch.tensor(base_obs, dtype=torch.float32)
        extras = {}
        extras["observations"]["actor"] = base_obs
        if self.use_privileged_obs:
            privileged_obs = torch.tensor(privileged_obs, dtype=torch.float32)
            extras['observations']['critic'] = privileged_obs
        
        return base_obs, extras
        
    def step(self, action):
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
                if self.days_since_last_trade < self.min_hold_days:
                    terminated = True
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
        
        # Check if the episode is done
        if self.current_timestep >= self.max_episode_length:
            truncated = True
        
        # Calculate the reward
        self.max_profit_since_buy = max(self.max_profit_since_buy, self.current_holding_value)
        reward = self._reward(action)
        reward = torch.tensor(reward, dtype=torch.float32)
        if self.action == 1:
            self.has_position != self.has_position
            self.current_holding_value = 0 # cleanup after buy/sell
            self.max_profit_since_buy = 0 # cleanup after buy/sell
        
        # Get the next observation
        observation, extras = self.get_observation()
        
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
        return observation, reward, dones, extras
    
    def _reward(self, action):
        """Calculates the reward based on the current state.
        The reward is calculated as the difference between the current net worth and the initial net worth.
        """
        
        #Greedy Reward: trying to maximize the current net worth through each step
        # if has_position try to maximize the current holding value and penalize if current holding is below max
        # if doesnt't have position, try to maximize loss 
        greedy_reward = 0
        if self.max_profit_since_buy != 0:
            greedy_reward = self.current_holding_value - self.max_profit_since_buy/ self.max_profit_since_buy
        else:
            greedy_reward = 0.1
        if not self.has_position:
            greedy_reward = -greedy_reward
        
        
        
        # Calculate the the current networth gain to if just held the stock
        holding = (self.current_close - self.inital_bought_price) / self.inital_bought_price
        holding_reward = (self.current_networth - holding) / holding
        
        #penalty for selling too early:
        min_hold_penalty = 0
        if self.days_since_last_trade < self.min_hold_days:
            min_hold_penalty = (self.days_since_last_trade - self.min_hold_days) / self.min_hold_days
        
        # consturct the reward
        
        
        reward = 0.5* greedy_reward + 0.5*holding_reward - 0.2*min_hold_penalty
        
        #clip the reward
        reward = np.clip(reward, -5, 5)
        
        return reward  
        
        
        
        
        
        
        
        
        
        
        
        
        