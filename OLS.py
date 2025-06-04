import numpy as np
from scipy.stats import t

def OLS(X_obs, Y_obs):
    #1. First, we need to append a column of ones to the X array to account for the y-intercept in the linear regression equation. 
    #So, we will   transform x into a 2-D array:
    # Reshape x to a 2-D array
    X_obs_m = X_obs.reshape(-1,1)
    
    # Append a column of ones
    X_obs_m = np.hstack((np.ones((X_obs_m.shape[0], 1)), X_obs_m))
    
    #2. Next, we calculate the matrix product of X transpose and X, and then the inverse of this product:
    X_transpose_dot_X = np.matmul(X_obs_m.T, X_obs_m)
    X_transpose_dot_X_inv = np.linalg.inv(X_transpose_dot_X)
    
    # 3. Then, we calculate the matrix product of X transpose and Y:
    X_transpose_dot_Y = np.matmul(X_obs_m.T, Y_obs)
    
    # 4. Finally, we can calculate the parameters theta by multiplying the two results from steps 2 and 3:
    theta = np.dot(X_transpose_dot_X_inv, X_transpose_dot_Y)
    
    return theta,  X_transpose_dot_X_inv#, X_obs_m



def confidence_int(X_obs, Y_obs, Y_hat, X_transpose_dot_X_inv, theta, sample, conf_level):
    #Reshape x to a 2-D array, append column of ones
    X_obs_m = X_obs.reshape(-1,1)
    X_obs_m = np.hstack((np.ones((X_obs_m.shape[0], 1)), X_obs_m))
    sample = sample.reshape(-1,1)
    sample = np.hstack((np.ones((sample.shape[0], 1)), sample))
    m2 = np.matmul(sample, X_transpose_dot_X_inv)
    m3 = np.diagonal(np.matmul(m2, sample.T))
    #print("m3:", m3.shape, m3[:5])
    
    #1. Compute the Residual Sum of Squares (RSS):
    Y_hat_obs = np.matmul(X_obs_m, theta)
    residuals = Y_obs - Y_hat_obs
    #print("residuals:", residuals[:5])
    RSS = np.sum(residuals**2)
    
    #2. Compute the standard error of the regression (SER):
    SER = np.sqrt(RSS / (len(Y_obs) - 2)) # n−k−1 degrees of freedom (with k number of regressors: 1 for simple linear regression #####         
    #print("SER", SER)

    y_pred_std_dev = np.sqrt((1 + m3) * np.square(SER))
    #print("y_pred_std_dev:", y_pred_std_dev.shape, y_pred_std_dev[:5])

    # Compute the critical value from the t-distribution
    alpha = 1 - conf_level
    dof = len(Y_obs) - 2  # degrees of freedom
    t_critical = t.ppf(1 - alpha/2, dof)
    #print("t:", t_critical)
    
    #The 95% confidence interval for the predicted values can be calculated similarly to the confidence intervals for theta:
    ci_lower_pred = Y_hat - t_critical * y_pred_std_dev
    ci_upper_pred = Y_hat + t_critical * y_pred_std_dev
        
    return ci_lower_pred, ci_upper_pred

