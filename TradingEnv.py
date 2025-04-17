import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd

import gymnasium as gym
import numpy as np
import pandas as pd

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
        randomize_episode (bool): Whether to randomize stock, start date, and shares at the beginning
                                 of each episode. Default: True.
        use_privileged_obs (bool): If True, includes the next 5 days' closing prices in the
                                   observation (intended for critic). Default: False.
        seed (int): Random seed for reproducibility. Default: 0.
        verbose (bool): If True, prints step action/reward information. Default: False.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self,
                 stock='AAPL',
                 initial_shares=3,
                 initial_buy_date="2023-12-01",
                 min_hold_days=10,
                 granularity='1d',
                 period='max',
                 randomize_episode=True,
                 use_privileged_obs=False,
                 seed=0,
                 verbose=False):
        super().__init__()

        # --- Configuration ---
        self.initial_stock = stock
        self.initial_shares_config = initial_shares
        self.initial_buy_date_config = initial_buy_date
        self.min_hold_days = min_hold_days
        self.granularity = granularity
        self.period = period
        self.randomize_episode = randomize_episode
        self.use_privileged_obs = use_privileged_obs
        self.seed_value = seed
        self.verbose = verbose # Controls step printing, not init/error prints
        self.rng = np.random.default_rng(self.seed_value)

        # --- Data Placeholders ---
        self.data = pd.DataFrame()
        self.vix_data = pd.DataFrame()
        self.gspc_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()

        # --- Randomization Options ---
        self.random_symbols = [
            "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL", "META", "ADBE",
            "NFLX", "VOO", "FTEC", "TSLA", "JPM", "V", "UNH"
        ]
        self.random_gran_period_pairs = {'1d': ['5y', '10y', 'max']}

        # --- Environment State ---
        self.current_stock = ""
        self.current_shares = 0
        self.initial_buy_date = ""
        self.current_step_index = 0
        self.initial_networth = 0.0
        self.current_networth = 0.0
        self.bought_price = 0.0
        self.current_holding_value = 0.0
        self.days_since_last_trade = 0
        self.has_position = False
        self.max_profit_since_buy = 0.0

        # --- Action Space ---
        self.action_space = gym.spaces.Discrete(2)

        # --- Observation Space ---
        obs_shape_base = 50
        obs_shape_total = obs_shape_base + 5 if self.use_privileged_obs else obs_shape_base
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape_total,), dtype=np.float32
        )

        # --- Initial Data Fetch ---
        # Fetch only if not randomizing each episode (randomize handles its own fetch)
        if not self.randomize_episode:
            # print(f"[INFO] Initializing environment for fixed stock: {self.initial_stock}")
            if not self._fetch_data(self.initial_stock, self.period, self.granularity):
                raise RuntimeError(f"Failed to fetch initial data for {self.initial_stock}. Check warnings/errors above.")
            if not self._set_initial_buy_date(self.initial_buy_date_config):
                raise ValueError(f"Initial buy date {self.initial_buy_date_config} invalid or lacks sufficient history in fetched data for {self.initial_stock}.")


    def _fetch_data(self, stock_symbol, period, interval):
        """Fetches and merges data, converts index to timezone-naive."""
        # print(f"[INFO] Fetching data for {stock_symbol}, ^VIX, ^GSPC (Period: {period}, Interval: {interval})")
        try:
            stock_ticker = yf.Ticker(stock_symbol)
            self.data = stock_ticker.history(period=period, interval=interval, auto_adjust=True)
            if self.data.empty: print(f"[WARNING] Failed to fetch data for primary stock {stock_symbol}.")

            vix_ticker = yf.Ticker("^VIX")
            self.vix_data = vix_ticker.history(period=period, interval=interval, auto_adjust=True)
            if self.vix_data.empty: print(f"[WARNING] Failed to fetch data for ^VIX.")

            gspc_ticker = yf.Ticker("^GSPC")
            self.gspc_data = gspc_ticker.history(period=period, interval=interval, auto_adjust=True)
            if self.gspc_data.empty: print(f"[WARNING] Failed to fetch data for ^GSPC.")

            stock_df = self.data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not self.data.empty else pd.DataFrame(index=self.data.index)
            vix_df = self.vix_data[['Close']].rename(columns={'Close': 'VIX_Close'}).copy() if not self.vix_data.empty else pd.DataFrame(index=self.vix_data.index)
            gspc_df = self.gspc_data[['Close']].rename(columns={'Close': 'GSPC_Close'}).copy() if not self.gspc_data.empty else pd.DataFrame(index=self.gspc_data.index)

            self.merged_data = pd.concat([stock_df, vix_df, gspc_df], axis=1, join='outer')
            if 'Close' not in self.merged_data.columns and not stock_df.empty:
                 print("[ERROR] Primary stock 'Close' column missing after outer join.")
                 # Let subsequent checks handle failure

            self.merged_data = self.merged_data.ffill(limit=5)
            self.merged_data.dropna(subset=['Close'], inplace=True) # Drop essential NaNs

            # Convert index to timezone-naive AFTER all data manipulation
            if pd.api.types.is_datetime64_any_dtype(self.merged_data.index) and self.merged_data.index.tz is not None:
                self.merged_data.index = self.merged_data.index.tz_localize(None)

            min_required_length = 35
            if len(self.merged_data) < min_required_length:
                print(f"[WARNING] Insufficient merged data after processing for {stock_symbol} (Final Length: {len(self.merged_data)}, Required: {min_required_length}).")
                return False

            # print(f"[INFO] Data fetched and processed successfully for {stock_symbol}. Final data length: {len(self.merged_data)}")
            return True

        except Exception as e:
            # print(f"[ERROR] Exception during data fetch/process for {stock_symbol}: {e}")
            self.merged_data = pd.DataFrame()
            return False

    def _set_initial_buy_date(self, requested_date_str):
        """Sets the initial buy date, finding the closest valid date if needed."""
        if self.merged_data.empty:
             print("[ERROR] Cannot set initial buy date: merged_data is empty.")
             return False
        try:
            requested_date = pd.to_datetime(requested_date_str).normalize() # Naive timestamp
            min_required_index_loc = 30

            # Compare naive requested_date with naive self.merged_data.index
            if requested_date in self.merged_data.index:
                date_index_loc = self.merged_data.index.get_loc(requested_date)
                if date_index_loc >= min_required_index_loc:
                    self.initial_buy_date = requested_date
                    return True # Found exact date with enough history
                # else: Date found but not enough history, search forward

            # Search forward if exact date not found or lacked history
            valid_dates_after_request = self.merged_data.index[self.merged_data.index >= requested_date]
            for valid_date in valid_dates_after_request:
                 if self.merged_data.index.get_loc(valid_date) >= min_required_index_loc:
                     self.initial_buy_date = valid_date
                    #  print(f"[INFO] Using closest valid date with sufficient history: {self.initial_buy_date.strftime('%Y-%m-%d')}")
                     return True # Found suitable date after requested

            print(f"[ERROR] Cannot find a suitable initial buy date >= {requested_date_str} with enough history ({min_required_index_loc} days).")
            return False

        except Exception as e:
            print(f"[ERROR] Error setting initial buy date for '{requested_date_str}': {e}")
            return False


    def _init_state(self):
        """Initializes the environment's state for the start of an episode."""
        if self.merged_data.empty: raise RuntimeError("Cannot initialize state: Data is not loaded.")
        if not isinstance(self.initial_buy_date, pd.Timestamp): raise RuntimeError("Cannot initialize state: Initial buy date is not set or invalid.")

        try:
            # Verify date still exists (it should if _set_initial_buy_date worked)
            self.current_step_index = self.merged_data.index.get_loc(self.initial_buy_date)
        except KeyError:
             print(f"[ERROR] Internal Error: Initial buy date {self.initial_buy_date} not found in final merged data index during state initialization.")
             # Attempt recovery? Or just raise error? Raising is safer.
             raise RuntimeError(f"Could not locate initial buy date {self.initial_buy_date} in final data index during init.")

        self.bought_price = self.merged_data.iloc[self.current_step_index]['Close']
        if pd.isna(self.bought_price):
             raise ValueError(f"Initial bought price is NaN at index {self.current_step_index} for date {self.initial_buy_date}. Check data processing.")

        self.current_shares = self.initial_shares_config
        self.initial_networth = self.current_shares * self.bought_price
        self.current_networth = self.initial_networth
        self.current_holding_value = 0.0
        self.days_since_last_trade = 0
        self.has_position = True
        self.max_profit_since_buy = 0.0


    def _randomize(self):
        """Randomizes parameters for a new episode."""
        # print("[INFO] Randomizing episode parameters...") # Can be noisy
        max_retries = 10
        for attempt in range(max_retries):
            self.granularity = self.rng.choice(list(self.random_gran_period_pairs.keys()))
            self.period = self.rng.choice(self.random_gran_period_pairs[self.granularity])
            self.current_stock = self.rng.choice(self.random_symbols)
            self.initial_shares_config = self.rng.integers(1, 11)

            if self._fetch_data(self.current_stock, self.period, self.granularity):
                min_start_idx_loc = 30
                required_future_steps = 5 if self.use_privileged_obs else 1
                max_start_idx_loc = len(self.merged_data) - 1 - required_future_steps

                if max_start_idx_loc >= min_start_idx_loc:
                    random_idx_loc = self.rng.integers(min_start_idx_loc, max_start_idx_loc + 1)
                    self.initial_buy_date = self.merged_data.index[random_idx_loc] # Already naive

                    if isinstance(self.initial_buy_date, pd.Timestamp):
                        #  print(f"[INFO] Randomization successful: Stock={self.current_stock}, Shares={self.initial_shares_config}, StartDate={self.initial_buy_date.strftime('%Y-%m-%d')}")
                        return # Success
                    else: print(f"[WARNING] Random index location {random_idx_loc} did not yield valid date type. Retrying...")
                else:
                    print(f"[WARNING] Not enough data range ({len(self.merged_data)} days) for randomization with {self.current_stock}. Retrying...")
            else:
                print(f"[WARNING] Data fetch/process failed during randomization for {self.current_stock}. Retrying...")

        raise RuntimeError(f"Failed to randomize episode after {max_retries} attempts.")


    def _calculate_momentum(self, lookback_days):
        """Calculates price momentum over a given lookback period."""
        if self.current_step_index < lookback_days: return 0.0
        try:
            past_price = self.merged_data.iloc[self.current_step_index - lookback_days]['Close']
            current_price = self.merged_data.iloc[self.current_step_index]['Close']
            if pd.isna(past_price) or pd.isna(current_price) or abs(past_price) < 1e-9: return 0.0
            return (current_price - past_price) / past_price
        except IndexError: return 0.0 # Expected if index is near start


    def _get_past_data(self, column, days):
        """Safely gets the past 'days' data for a given column, padding and filling NaNs."""
        if self.current_step_index < days:
            past_data = self.merged_data.get(column, pd.Series(dtype=float)).iloc[:self.current_step_index].values
            padding_value = past_data[0] if len(past_data) > 0 and not pd.isna(past_data[0]) else 0.0
            padding_size = days - len(past_data)
            padded_data = np.pad(past_data.astype(float), (padding_size, 0), 'constant', constant_values=padding_value)
            return np.nan_to_num(padded_data.flatten(), nan=0.0)
        else:
            data_slice = self.merged_data.get(column, pd.Series(dtype=float)).iloc[self.current_step_index - days : self.current_step_index].values.flatten()
            return np.nan_to_num(data_slice, nan=0.0)


    def _get_future_data(self, column, days):
        """Safely gets the future 'days' data, padding and filling NaNs."""
        start_index = self.current_step_index + 1
        end_index = start_index + days
        max_index = len(self.merged_data)

        if end_index <= max_index:
            data_slice = self.merged_data.get(column, pd.Series(dtype=float)).iloc[start_index : end_index].values.flatten()
            return np.nan_to_num(data_slice, nan=0.0)
        else:
            future_data = self.merged_data.get(column, pd.Series(dtype=float)).iloc[start_index:max_index].values
            padding_size = days - len(future_data)
            if len(future_data) > 0 and not pd.isna(future_data[-1]): padding_value = future_data[-1]
            else:
                current_val = self.merged_data.get(column, pd.Series(dtype=float)).iloc[self.current_step_index]
                padding_value = current_val if not pd.isna(current_val) else 0.0
            padded_data = np.pad(future_data.astype(float), (0, padding_size), 'constant', constant_values=padding_value)
            return np.nan_to_num(padded_data.flatten(), nan=0.0)


    def _get_observation(self):
        """Constructs the observation vector for the current state."""
        if self.merged_data.empty or not (0 <= self.current_step_index < len(self.merged_data)):
             # This case should ideally be prevented by checks in step/reset
             print("[ERROR] _get_observation called with invalid state/index.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_data = self.merged_data.iloc[self.current_step_index]

        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        ohlc = np.nan_to_num(np.array([current_data.get(col, 0.0) for col in ohlc_cols], dtype=np.float32), nan=0.0)
        volume = np.nan_to_num(np.array([current_data.get('Volume', 0.0)], dtype=np.float32), nan=0.0)
        past_30d_close = self._get_past_data('Close', 30)
        past_5d_vix = self._get_past_data('VIX_Close', 5)
        past_5d_gspc = self._get_past_data('GSPC_Close', 5)
        momentum_7d = np.array([self._calculate_momentum(7)], dtype=np.float32)
        momentum_30d = np.array([self._calculate_momentum(30)], dtype=np.float32)
        holding_flag = np.array([1.0 if self.has_position else 0.0], dtype=np.float32)
        current_close = current_data.get('Close', 0.0)
        if self.has_position and not pd.isna(current_close) and abs(current_close) > 1e-6:
             relative_bought_price = np.array([(self.bought_price - current_close) / current_close], dtype=np.float32)
        else: relative_bought_price = np.array([0.0], dtype=np.float32)
        relative_bought_price = np.nan_to_num(relative_bought_price, nan=0.0)
        days_since_trade = np.array([self.days_since_last_trade], dtype=np.float32)

        try:
            observation_list = [
                ohlc, volume, past_30d_close, past_5d_vix, past_5d_gspc,
                momentum_7d, momentum_30d, holding_flag, relative_bought_price,
                days_since_trade
            ]
            observation = np.concatenate(observation_list).astype(np.float32)
        except ValueError as e:
            print(f"[ERROR] Concatenation failed in _get_observation: {e}")
            shapes = [arr.shape for arr in observation_list]; print(f"Shapes: {shapes}")
            expected_base_len = self.observation_space.shape[0] - (5 if self.use_privileged_obs else 0)
            observation = np.zeros(expected_base_len, dtype=np.float32) # Fallback

        if self.use_privileged_obs:
            next_5d_close = self._get_future_data('Close', 5)
            try: observation = np.concatenate([observation, next_5d_close]).astype(np.float32)
            except ValueError as e:
                 print(f"[ERROR] Concatenation failed for privileged features: {e}")
                 expected_len = self.observation_space.shape[0]; current_len = len(observation)
                 if current_len < expected_len: observation = np.concatenate([observation, np.zeros(expected_len - current_len, dtype=np.float32)])
                 elif current_len > expected_len: observation = observation[:expected_len]

        expected_len = self.observation_space.shape[0]
        if len(observation) != expected_len:
             print(f"[ERROR] Observation length mismatch! Expected {expected_len}, got {len(observation)}. Padding/Truncating.")
             if len(observation) < expected_len: observation = np.concatenate([observation, np.zeros(expected_len - len(observation), dtype=np.float32)])
             else: observation = observation[:expected_len]
        if np.isnan(observation).any():
             print(f"[WARNING] NaN detected in final observation at step {self.current_step_index}. Replacing with 0.")
             observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        return observation


    def _reward(self, action):
        """
        Revised Reward Structure with Negative Outcomes:
        - When holding:
            Reward = portfolio return (current price vs. bought price)
                    minus a time penalty 
                    minus a penalty proportional to the missed opportunity
                    (i.e. how far the current unrealized profit is below the maximum observed).
        - When not holding:
            Reward = realized return (on networth) plus a term for the potential return
                    (what the return would be if still holding) minus a small time penalty.
        """
        # Check valid index and current price
        if not (0 <= self.current_step_index < len(self.merged_data)):
            return 0.0

        current_close = self.merged_data.iloc[self.current_step_index]['Close']
        if pd.isna(current_close):
            return 0.0

        # Define a time penalty for each day held
        time_penalty = 0.001 * self.days_since_last_trade

        # Weight factors (tune these as needed)
        greedy_weight = 0.5    # Penalty for missed opportunity when holding
        potential_weight = 0.5 # Weight for potential market return when not holding

        if self.has_position:
            # Calculate portfolio return as percentage change from buy price
            portfolio_return = (current_close - self.bought_price) / self.bought_price
            
            # Compute missed opportunity: difference between maximum unrealized profit and current profit, as a percentage of initial cost
            cost = self.bought_price * self.current_shares
            if cost == 0:
                missed_profit_pct = 0.0
            else:
                missed_profit_pct = (self.max_profit_since_buy - self.current_holding_value) / cost
            
            # The reward is the actual return minus the time penalty and missed opportunity penalty.
            reward = portfolio_return - time_penalty - greedy_weight * missed_profit_pct

        else:
            # When not holding, calculate realized return from networth change
            realized_return = (self.current_networth - self.initial_networth) / self.initial_networth
            
            # And consider potential market return if you had stayed in (this term can be negative)
            potential_return = (current_close - self.bought_price) / self.bought_price
            
            # Combine realized and potential returns, then subtract a small time penalty.
            reward = realized_return + potential_weight * potential_return - 0.1 * time_penalty

        # For debugging, return the raw reward value to see negative values when appropriate.
        return reward


    def step(self, action):
        if self.merged_data.empty or not (0 <= self.current_step_index < len(self.merged_data)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, True, {"message": "Step called in invalid state."}

        reward = self._reward(action)  # Calculated based on current state before action
        action_taken = "Hold"
        current_price = self.merged_data.iloc[self.current_step_index]['Close']

        # Apply action logic
        if pd.isna(current_price):
            action = 0
            self.days_since_last_trade += 1
            action_taken = "Hold (Forced due to NaN price)"
        elif action == 1:
            self.days_since_last_trade = 0
            self.max_profit_since_buy = 0.0
            self.current_holding_value = 0.0
            if self.has_position:  # Sell
                # Use the original bought_price to calculate profit
                profit = self.current_shares * (current_price - self.bought_price)
                self.current_networth += profit
                self.has_position = False 
                self.bought_price = current_price
                action_taken = "Sell"
            else:  # Buy
                self.current_shares = self.initial_shares_config
                self.bought_price = current_price  # Set bought price here for a new position
                self.has_position = True
                action_taken = "Buy"
        else:  # Hold
            self.days_since_last_trade += 1
            if self.has_position:
                unrealized_gain = self.current_shares * (current_price - self.bought_price)
                self.current_holding_value = unrealized_gain
                self.max_profit_since_buy = max(self.max_profit_since_buy, unrealized_gain)
                action_taken = "Hold (Position)"
            else:
                action_taken = "Hold (No Position)"

        # Advance time AFTER applying action and calculating reward for current state
        self.current_step_index += 1

        # Determine next state, termination, truncation
        terminated, truncated = False, False
        observation = None
        # if self.min_hold_days > 0 and self.days_since_last_trade < self.min_hold_days:
        #     terminated = True
        # print(f"[INFO] Minimum hold days not met. Current: {self.days_since_last_trade}, Required: {self.min_hold_days}")
        # Compute portfolio value based on whether a position is held or not:
        if self.current_step_index < len(self.merged_data):
            current_price_next_day = self.merged_data.iloc[self.current_step_index]['Close']
            if self.has_position and not pd.isna(current_price_next_day):
                current_total_value = self.current_shares * current_price_next_day
            else:
                current_total_value = self.current_networth
            current_price_final = current_price_next_day

            required_future_steps = 5 if self.use_privileged_obs else 1
            if self.current_step_index >= len(self.merged_data) - required_future_steps:
                truncated = True  # Truncate if not enough future data for next obs

            observation = self._get_observation()
        else:
            truncated = True
            last_valid_index = self.current_step_index - 1
            self.current_step_index = last_valid_index  # Revert for final obs calc
            observation = self._get_observation()
            self.current_step_index += 1  # Restore index
            current_price_final = self.merged_data.iloc[last_valid_index]['Close']
            current_total_value = self.current_networth

        # Gather Info dictionary
        info = {
            "timestamp": self.merged_data.index[self.current_step_index] if self.current_step_index < len(self.merged_data) else self.merged_data.index[-1],
            "stock": self.current_stock,
            "action_taken": action_taken,
            "start_networth": self.initial_networth, 
            "current_total_value": current_total_value,
            "profit_loss": current_total_value - self.initial_networth,
            "profit_loss_pct": (current_total_value - self.initial_networth) / self.initial_networth if abs(self.initial_networth) > 1e-6 else 0.0,
            "current_price": np.nan_to_num(current_price_final, nan=0.0),
            "has_position": self.has_position,
            "days_held": self.days_since_last_trade if self.has_position else 0,
            "reward": reward,
        }
        if observation is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        if self.verbose and not (terminated or truncated):
            print(f"Step: {self.current_step_index} \n"
                f"Date: {info['timestamp'].strftime('%Y-%m-%d')} \n"
                f"Action: {info['action_taken']} \n"
                f"Next Price: {info['current_price']:.2f} \n"
                f"Position: {info['has_position']} \n"
                f"Total Value: {info['current_total_value']:.2f} \n"
                f"Reward: {reward:.4f} \n" )

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
             self.seed_value = seed
             self.rng = np.random.default_rng(self.seed_value)
             # print(f"[INFO] Resetting environment with seed: {self.seed_value}") # Can be noisy

        if self.randomize_episode:
            self._randomize() # Handles fetching & setting date
        else:
            # Fetch only if not already done (e.g., first reset)
            if self.merged_data.empty:
                #  print(f"[INFO] First reset for fixed stock {self.initial_stock}, fetching data...")
                 if not self._fetch_data(self.initial_stock, self.period, self.granularity):
                      raise RuntimeError(f"Failed to fetch data for {self.initial_stock} during reset.")
            # else: # Data already exists, just ensure stock name is set
            #     #  print(f"[INFO] Resetting fixed environment: {self.initial_stock}")
            self.current_stock = self.initial_stock
            # Set/validate the initial buy date
            if not self._set_initial_buy_date(self.initial_buy_date_config):
                 # Attempt fallback if setting specific date fails
                 min_required_index_loc = 30
                 if len(self.merged_data) > min_required_index_loc + 5: # Need buffer
                      fallback_date = self.merged_data.index[min_required_index_loc]
                      print(f"[WARNING] Initial buy date '{self.initial_buy_date_config}' invalid. Using fallback: {fallback_date.strftime('%Y-%m-%d')}")
                      self.initial_buy_date = fallback_date
                 else:
                      raise ValueError("Cannot reset environment. No valid initial buy date found and insufficient data for fallback.")

        # Initialize state based on potentially randomized parameters
        self._init_state()

        observation = self._get_observation()
        info = {
            "message": "Environment reset",
            "initial_networth": self.initial_networth,
            "initial_buy_date": self.initial_buy_date.strftime('%Y-%m-%d') if isinstance(self.initial_buy_date, pd.Timestamp) else "N/A",
            "stock": self.current_stock
        }
        # print(f"[INFO] Reset complete.") # Can be noisy
        return observation, info

    def render(self, mode='human'):
        """Renders the environment state information."""
        # Simplified render, primarily for debugging, not needed for training
        if mode == 'human':
            render_index = self.current_step_index
            if render_index >= len(self.merged_data): render_index = len(self.merged_data) - 1
            if render_index < 0: return

            print("-" * 30); print(f"Render State at Step: {render_index}")
            current_timestamp = self.merged_data.index[render_index]
            print(f"Date: {current_timestamp.strftime('%Y-%m-%d')}")
            print(f"Stock: {self.current_stock}")
            current_price = self.merged_data.iloc[render_index]['Close']
            print(f"Price: {current_price:.2f}" if not pd.isna(current_price) else "Price: N/A")
            print(f"Position: {'Holding' if self.has_position else 'None'}")
            # Add more details if needed for debugging
            print("-" * 30)
        else: return super(TradingEnv, self).render(mode=mode)

    def close(self):
        """Performs any necessary cleanup."""
        print("[INFO] Closing Trading Environment.")
        self.data = pd.DataFrame(); self.vix_data = pd.DataFrame()
        self.gspc_data = pd.DataFrame(); self.merged_data = pd.DataFrame()
        pass
