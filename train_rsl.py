import numpy as np
import os.path
import yaml
import time
import torch
from rsl_rl.runners import OnPolicyRunner
from RSLTradingEnv import TradingEnv


env = TradingEnv()

# Load the configuration file
