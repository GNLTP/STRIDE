import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Gauss_FFT_for_Period(x, k=3, sigma=0.2):

    B, L, N = x.shape
    xf = torch.fft.rfft(x, dim=1)  

    frequency_list = abs(xf).mean(0).mean(-1).detach().cpu().numpy()  
    freqs = torch.fft.rfftfreq(L, d=1.0).cpu().numpy()                

    if not np.isfinite(frequency_list).all():
        raise RuntimeError("[Period_init] frequency_list has NaN/Inf")

    freqs = freqs[1:]
    frequency_list = frequency_list[1:]

    unique_periods = []             
    seen = set()
    freq_mask = np.ones_like(freqs) 

    max_period = freqs.shape[0]
    log_p = np.log(1.0 / freqs)
    log_min, log_max = log_p.min(), log_p.max()

    for _ in range(k):
        masked_energy = frequency_list * freq_mask
        idx = np.argmax(masked_energy)  
        f = freqs[idx]
        if f <= 0:
            continue

        mapped_p = 2 + (log_p[idx] - log_min) / (log_max - log_min) * (max_period - 2)
        period = int(round(mapped_p))

        if period not in seen:
            seen.add(period)
            unique_periods.append(period)

            gaussian = np.exp(-((freqs - f) ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.max()
            freq_mask *= (1 - gaussian)  

        if len(unique_periods) >= k:
            break

    while len(unique_periods) < k:
        unique_periods.append(np.random.choice(unique_periods))

    return unique_periods

def FFT_for_Period(x, k=3, tolerance=1e-3):

    B, L, N = x.shape
    xf = torch.fft.rfft(x, dim=1)  

    frequency_list = abs(xf).mean(0).mean(-1).detach().cpu().numpy()  
    freqs = torch.fft.rfftfreq(L, d=1.0).cpu().numpy()                

    freqs = freqs[1:]
    frequency_list = frequency_list[1:]

    sorted_indices = np.argsort(frequency_list)[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_periods = [int(round(1.0 / f)) for f in sorted_freqs if f > 0]

    seen = set()
    unique_periods = []
    for p in sorted_periods:
        if p not in seen:
            seen.add(p)
            unique_periods.append(p)
        if len(unique_periods) >= k:
            break

    while len(unique_periods) < k:
        unique_periods.append(np.random.choice(unique_periods))

    return unique_periods

def Period_init(x, k=3):
    _, T, _ = x.size()

    period_list = Gauss_FFT_for_Period(x, k=k, sigma=0.2)

    period_init = []
    for i in range(k):
        period = int(period_list[i])  
        length = ((T + period - 1) // period) * period  

        H = length // period
        W = period
        period_init.append([length, H, W])

    return period_init
