import gymnasium as gym
import numpy as np
import yfinance as yf

class TradingEnv(gym.Env):
    def __init__(self, stock='AAPL', minbuy=10, gran='1d', period='max', shares=3, boughtat="2023-12-01", randomize=True, seed=0, verbose=False):
        super(TradingEnv, self).__init__()
        self.running_mean = 0
        self.running_std = 1  
        self.decay_factor = 0.99  
        self.epsilon = 1e-7  
        
        self.stock = stock
        self.minbuy = minbuy
        self.gran = gran
        self.period = period
        self.shares = shares
        self.boughtat = boughtat
        self.seed = seed
        self.randomize = randomize
        self.max_networth = 0
        self.print = verbose
        self.streak = 0
        self.is_init = True
        
        if self.gran in ['5m', '15m', '30m']:
            print("[WARNING] Data granularity can only go back to 60 Days Maxiumum")
        if self.gran in ['1h', '60m']:
            print("[WARNING] Data granularity can only go back to 730 Days Maxiumum")
        self.random_symbols = [
            "NVDA" , "AMZN", "GOOGL", "MSFT", "AAPL", "META", "NVDA", "ADBE", "NFLX", "NANC", "KRUZ", "VOO", "FTEC", "TSLA"
        ]
        self.random_gran_period_pairs = {
            '1d' : ['max']
        }
        if self.randomize:
            self._randomize()
        else:
            self.data = yf.Ticker(self.stock).history(period=self.period, interval=self.gran, auto_adjust=True)
        
        self._init_obs()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 48), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.networth_rewd = 0
        
    def _init_obs(self):
        self.current_day = self.boughtat
        self.cur_indx = self.data.index.get_loc(self.current_day)
        self.init_networth = self.shares * self.data.loc[self.boughtat, 'Close']
        self.networth = self.init_networth
        self.previous_open_close = self.data.loc[self.boughtat, 'Open'], self.data.loc[self.boughtat, 'Close']
        self.previous_high_low = self.data.loc[self.boughtat, 'High'], self.data.loc[self.boughtat, 'Low']
        self.previous_volume = self.data.loc[self.boughtat, 'Volume']
        self.previous_momentum_wk = self._calculate_momentum(7)
        self.previous_momentum_mon = self._calculate_momentum(30)
        self.previous_wk_close = np.zeros((1,7))
        self.previous_mo_close = np.zeros((1,30))
        self.bought_price = self.data.loc[self.boughtat, 'Close']
        self.current_shares = self.shares
        self.relneworth = 0
        self.current_action = 0
        self.max_networth = 0
        self.streak = 0
        self.is_init = True

        self.has_position = True
        self.previous_obs = np.zeros((1, 48)).astype(np.float32)
        
        
    def _randomize(self):
        self.random_seed = np.random.randint(0, 100)
        np.random.seed(self.random_seed)
        # Randomly select number of shares from 1 to 10
        self.shares = np.random.randint(1, 10)
        
        # Randomly select granularity and period
        while True:
            self.gran = np.random.choice(list(self.random_gran_period_pairs.keys()))
            self.period = np.random.choice(self.random_gran_period_pairs[self.gran])
            
            # Randomly select a stock
            self.stock = np.random.choice(self.random_symbols)
            
            try:
                # Fetch data for the randomized stock, period, and granularity
                self.data = yf.Ticker(self.stock).history(period=self.period, interval=self.gran, auto_adjust=True)
                self.boughtat = np.random.choice(self.data.index[31:])
                break
            except:
                continue
    
    def reset(self, seed=0):
        super(TradingEnv, self).reset()
        if self.randomize:
            self._randomize()
        else:
            self.data = yf.Ticker(self.stock).history(period=self.period, interval=self.gran, auto_adjust=True)
        self.seed = seed
        self._init_obs()
        info = {
            "init_networth": self.init_networth,
            "current_networth": self.networth,
            "current_action": self.current_action,
            "current_gain": self.relneworth,
            "diff": self.networth - self.init_networth,
            "diff_norm" : (self.networth - self.init_networth) / self.init_networth,
            "reward": self._reward(self.current_action),
            
        }
        return self.previous_obs, info
    
    def _calculate_momentum(self, interval):
        if self.cur_indx - interval < 0:
            return 0  # Avoid accessing invalid index
        past_price = self.data.iloc[self.cur_indx - interval]['Close']
        current_price = self.data.iloc[self.cur_indx]['Close']
        return (current_price - past_price) / past_price

    
    def _get_previous_close(self, interval):
        return self.data.loc[self.data.index[self.cur_indx-interval:self.cur_indx], 'Close'].to_numpy().reshape(1, interval)
    
    def step(self, action):
        if action == 1:
            if self.has_position:
                self.relneworth = self.current_shares * (self.previous_open_close[1] - self.bought_price)
                self.has_position = False
                self.networth += self.relneworth
            else:
                self.has_position = True
                self.relneworth = 0
            self.streak = 0
            self.bought_price = self.previous_open_close[1]
            self.current_action = 1
            self.max_networth = 0
        else:
            self.streak += 1
            self.current_action = 0
            self.relneworth = self.shares * (self.previous_open_close[1] - self.bought_price)
            if self.relneworth > self.max_networth:
                self.max_networth = self.relneworth
            
        self.cur_indx += 1
        info = {
            "init_networth": self.init_networth,
            "current_networth": self.networth,
            "current_action": self.current_action,
            "current_gain": self.relneworth,
            "diff": self.networth - self.init_networth,
            "diff_norm" : (self.networth - self.init_networth) / self.init_networth,
            "reward": self._reward(self.current_action),
        }
        try:
            self.current_day = self.data.index[self.cur_indx]
            terminated = False
            truncated = False
        except IndexError:
            print("[INFO] Data Exhausted")
            self.current_day = self.data.index[-1]
            terminated = False
            truncated = True
            return (
                self.prev_obs,
                self._reward(action),
                True,
                True,
                info
            )

        self.previous_open_close = np.array([self.data.loc[self.current_day, 'Open'], self.data.loc[self.current_day, 'Close']])
        self.previous_high_low = np.array([self.data.loc[self.current_day, 'High'], self.data.loc[self.current_day, 'Low']])
        self.previous_volume = np.array([self.data.loc[self.current_day, 'Volume']])
        self.previous_momentum_wk = self._calculate_momentum(7)
        self.previous_momentum_mon = self._calculate_momentum(30)
        self.previous_wk_close = self.data.loc[self.data.index[self.cur_indx-7:self.cur_indx], 'Close'].to_numpy()
        self.previous_mo_close = self.data.loc[self.data.index[self.cur_indx-30:self.cur_indx], 'Close'].to_numpy()

        # Flatten and construct the observation array
        observation = np.concatenate([
            self.previous_open_close.flatten(), # 2
            self.previous_high_low.flatten(), # 2
            self.previous_volume.flatten(), # 1
            self.previous_wk_close.flatten(), # 7
            self.previous_mo_close.flatten(), # 30
            np.array([
                self.previous_momentum_wk, # 1
                self.previous_momentum_mon, # 1
                self.bought_price, # 1
                self.current_shares, # 1
                self.relneworth, # 1
                self.current_action # 1
            ])
        ]).astype(np.float32)
        
        observation = observation.reshape(1, -1)
        self.prev_obs = observation

        # Termination terms
        if self.networth < (self.init_networth * 0.90):
            # print("[INFO] Terminated due to Loss")
            terminated = True
        truncated = False
        if self.cur_indx >= len(self.data.index) - 1:
            truncated = True


        # Reward
        reward = self._reward(action)
        
        self.is_init = False

        # Additional Info
        if self.print:
            print(" ")
            print("---------------------------------------------------------")
            print(f"[INFO] Current Stock: {self.stock}")
            print(f"[INFO] Current Day: {self.current_day}")
            print(f"[INFO] Current Open: {self.previous_open_close[0]}")
            print(f"[INFO] Current Net Worth: {self.networth}")
            print(f"[INFO] Holding Share: {self.has_position}")
            print(f"[INFO] Current Share Net Worth: {self.relneworth}")
            print("[INFO] Current Action: " + ("Hold" if self.current_action == 0 else "Sell" if self.current_action == 1 and not self.has_position else "Buy"))
            print("---------------------------------------------------------")

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    def _reward(self, action):
        reward_weights = {
            "relnetworth": 0.7,
            "streak": 0.3,
            
        }
        
        self.networth_rewd, self.streak_rewd = 0, 0
        
        # self.networth_rewd = self.relneworth
        if self.has_position:
            if self.relneworth < self.max_networth and self.max_networth > 0:
                self.networth_rewd = (self.relneworth - self.max_networth)/self.max_networth
            else:
                self.networth_rewd = self.relneworth / self.bought_price
        else:
            self.networth_rewd = (self.relneworth / self.bought_price) * -1
        
        if (self.streak < self.minbuy and not self.is_init) and action == 1:
            self.streak_rewd = -10
        else:
            self.streak_rewd = 2
        
        # Optionally clip the normalized reward to avoid very large values
        reward_min = -7
        reward_max = 7
        
        normalized_reward = reward_weights["relnetworth"] * self.networth_rewd + reward_weights["streak"] * self.streak_rewd
        normalized_reward = np.clip(normalized_reward, reward_min, reward_max)
        
        return normalized_reward
    
    def render(self):
        # Graph out the stock price with the buy and sell points
        pass
    
    def close(self):
        pass