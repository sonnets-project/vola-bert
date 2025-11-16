import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def tuple_unsqueeze(x):
    device = "cuda"
    return tuple([x[0].unsqueeze(0).to(device), {tok_name: tok_val.unsqueeze(0).to(device) for tok_name, tok_val in x[1].items()}])

def inv_transform(volas):
    _mean, _std = (0.0012177070210652764, 0.0009884204622900455)
    return volas * _std + _mean

def plot_predictions(ax, past_vals, future_vals, pred1, pred2, title=None, xticks_label=None):

    # converts all to np arrays
    if not isinstance(past_vals, np.ndarray): past_vals = np.array(past_vals)
    if not isinstance(future_vals, np.ndarray):  future_vals = np.array(future_vals)
    if not isinstance(pred1, np.ndarray):  pred1 = np.array(pred1)
    if not isinstance(pred2, np.ndarray):  pred2 = np.array(pred2)
    

    x = np.arange(len(past_vals) + len(future_vals))
    shade_start = len(past_vals)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.plot(x, np.concatenate([past_vals, future_vals]), color='black', linewidth=1)
    
    
    ax.plot(x[-len(pred1)-1:], np.concatenate([past_vals[-1:], pred1]), color='blue', linewidth=1, label="Baseline Prediction")
    ax.plot(x[-len(pred2)-1:], np.concatenate([past_vals[-1:], pred2]), color='red', linewidth=1, label="Counterfactual Prediction")
    ax.axvspan(shade_start, x[-1], color='red', alpha=0.15)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    
    if xticks_label:
        ax.set_xticks([len(past_vals)], [xticks_label], ha="right")

    if title:
        ax.set_title(title, fontsize=10)