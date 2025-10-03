import numpy as np 
import matplotlib.pyplot as plt

d = 50   #feature dimension

# Load the dataset
X = np.load('p3/data/X.npy')
y = np.load('p3/data/y.npy')
print ("data shape: ", X.shape, y.shape)

theta_star = np.load('p3/data/theta_star.npy')

###### part (1): least square estimator ########

# Calculate the least square solution
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

Error_LS = np.linalg.norm(theta_hat - theta_star, 2)
print('Estimator approximated by LS:',Error_LS)

###### part (2): L1 estimator ########
mu = 1e-5  # smoothing parameter
alpha = 0.001  # stepsize
T = 1000  # iteration number

# random initialization
theta = np.random.randn(d,1)

Error_huber = []

for _ in range(1, T):

    # calculate the l2 error of the current iteration
    Error_huber.append(np.linalg.norm(theta-theta_star, 2)) 

    # Calculate the residual
    r = y - X @ theta
    
    # Calculate the components of the gradient based on Huber loss
    g = np.where(np.abs(r) <= mu, r / mu, np.sign(r))
    
    # Calculate the final gradient
    grad = -X.T @ g

    #gradient descent update
    theta = theta - alpha * grad
    
#######   plot the figure   #########
plt.figure(figsize=(10,5))
plt.yscale('log',base=2) 
plt.plot(Error_huber, 'b-')
plt.title(r'$\ell_1$ estimator approximated by Huber')
plt.ylabel(r'$||\theta - \theta^*||_2$')
plt.xlabel('Iteration')
# plt.grid(True)
plt.show()