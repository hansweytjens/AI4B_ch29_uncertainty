import numpy as np

def generator(N_obs, N_unk, X_range, sigma, heterosc):
    # compute observed samples
    X_obs = np.random.uniform(X_range[0],X_range[1],N_obs)
    noise =  np.random.normal(0, sigma, N_obs)
    if heterosc:
        noise *= (X_obs - X_range[0])
    else:
        noise *= (X_range[1] - X_range[0]) / 2
    Y_obs = 10 + .2 * X_obs + noise
    Y_obs = np.maximum(Y_obs, 0)
    # compute unknown samples
    X_unk = np.random.uniform(X_range[1],X_range[2],N_unk)
    noise =  np.random.normal(0, sigma, N_unk)
    if heterosc:
        noise *= (X_unk - X_range[0])
    else:
        noise *= (X_range[1] - X_range[0]) / 2
    Y_unk = 10 + .2 * X_range[1] + noise
    Y_unk = np.maximum(Y_unk, 0)
    # return samples    
    return X_obs, Y_obs, X_unk, Y_unk