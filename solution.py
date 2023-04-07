import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 1371486987 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    max_x = np.max(x)
    a = 0.074
    a_hat = np.min(x)
    s = np.std(x, ddof=1)
    
    alpha = 1 - p
    loc = x.mean()
    scale = np.sqrt(np.var(x)) / np.sqrt(len(x))
    return loc - scale * norm.ppf(1 - alpha / 2), \
           loc - scale * norm.ppf(alpha / 2)
