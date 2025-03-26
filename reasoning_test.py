import numpy as np
import matplotlib.pyplot as plt

def steer_activations(x, decoder_weight, feature_index, amax, strength=2.0):
    d_i = decoder_weight[:, feature_index]
    return x + strength * amax * d_i

def plot_feature_activations(scores, top_k=20):
    top_indices = np.argsort(scores)[-top_k:]
    plt.barh(range(top_k), scores[top_indices])
    plt.yticks(range(top_k), [f'Feature {i}' for i in top_indices])
    plt.xlabel('ReasonScore')
    plt.title('Top Reasoning Features')
    plt.show()

def compute_reason_score(feature_activations, tokens_r, tokens_not_r, alpha=0.7):
    mu_r = feature_activations[tokens_r].mean(axis=0)
    mu_not_r = feature_activations[tokens_not_r].mean(axis=0)

    norm_mu_r = mu_r / mu_r.sum()
    norm_mu_not_r = mu_not_r / mu_not_r.sum()

    pi_r = feature_activations[tokens_r] / feature_activations[tokens_r].sum(axis=0, keepdims=True)
    entropy = -np.sum(pi_r * np.log(np.clip(pi_r, 1e-8, 1)), axis=0) / np.log(len(tokens_r))
    
    return (norm_mu_r * (entropy**alpha)) - norm_mu_not_r
