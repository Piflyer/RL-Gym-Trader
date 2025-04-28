from RSLTradingEnv import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
from tqdm import tqdm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import ConfigParser, Dataloader, PadPrivilegedObsWrapper
import requests_cache
import datetime
from time import time
writer = SummaryWriter()

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



average_reward = []
average_episode_length = []
average_gain = []
average_inference_time = []
for key in tqdm(dataloader.keys()):
    for j in range(3):
        data = {key: dataloader[key].iloc[-(300*(j+1) + 40):-(300*j + 1)]}
        eval_env_kwargs = {
        "dataloader": data,
        "max_episode_length": configPasrer.get("eval_steps", 300),
        "extra_obs": configPasrer.get("extra_obs", True),
        "device": configPasrer.get("device", "cpu"),
        "rl_platform": configPasrer.get("rl_platform", "SB3"),
        "period": configPasrer.get("period", "max"),
        "use_privileged_obs": False,
        "min_percnt": configPasrer.get("min_percnt", 0.8),
        "eval": True,
        "eval_buffer": 0,
        "failed_trade_terminate": configPasrer.get("failed_trade_terminate", True),
        }
        env = TradingEnv(**eval_env_kwargs)
        if configPasrer.get("use_privileged_obs"):
            env  = PadPrivilegedObsWrapper(env, priv_dim=18)
        model = PPO.load(configPasrer.get("eval_model"), env=env, device=configPasrer.get("device", "cpu"))
        env = model.get_env()
        obs = env.reset()
        len_steper = 0
        for i in range(300):
            start = time()
            action, _states = model.predict(obs, deterministic=True)
            end = time()
            average_inference_time.append(end - start)
            obs, reward, dones, extras = env.step(action)
            len_steper += 1
            average_reward.append(reward)
            if dones:
                average_gain.append(extras[0]["net_gain"])
                average_episode_length.append(len_steper)
                break
    
average_reward = np.array(average_reward)
average_episode_length = np.array(average_episode_length)
average_gain = np.array(average_gain)
average_inference_time = np.array(average_inference_time)

print("Model Name: ", configPasrer.get("eval_model"))
print("Average reward: ", np.mean(average_reward))
print("Average episode length: ", np.mean(average_episode_length))
print("Maximum episode length: ", np.max(average_episode_length))
print("Minimum episode length: ", np.min(average_episode_length))
print("Average gain: ", np.mean(average_gain))
print("Maximum gain: ", np.max(average_gain))
print("Minimum gain: ", np.min(average_gain))
print("Average inference time: ", np.mean(average_inference_time))
        
            
                
            
            
            
    

