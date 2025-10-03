import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_mat(path, d=16):
    data = scipy.io.loadmat(path)['zip']
    size = data.shape[0]
    y = data[:, 0].astype('int')
    X = data[:, 1:].reshape(size, d, d)
    return X, y

def cal_intensity(X):
    """
    X: (n, d, d), input data
    return intensity: (n, 1)
    """
    n = X.shape[0]
    return np.mean(X.reshape(n, -1), 1, keepdims=True)

def cal_symmetry(X):
    """
    X: (n, d, d), input data
    return symmetry: (n, 1)
    """
    n, d = X.shape[:2]
    Xl = X[:, :, :int(d/2)]
    Xr = np.flip(X[:, :, int(d/2):], -1)
    abs_diff = np.abs(Xl-Xr)
    return np.mean(abs_diff.reshape(n, -1), 1, keepdims=True)

def cal_feature(data):
    intensity = cal_intensity(data)
    symmetry = cal_symmetry(data)
    feat = np.hstack([intensity, symmetry])
    return feat

def cal_feature_cls(data, label, cls_A=1, cls_B=5):
    """ calculate the intensity and symmetry feature of given classes
    Input:
        data: (n, d1, d2), the image data matrix
        label: (n, ), corresponding label
        cls_A: int, the first digit class
        cls_B: int, the second digit class
    Output:
        X: (n', 2), the intensity and symmetry feature corresponding to
            class A and class B, where n'= cls_A# + cls_B#.
        y: (n', ), the corresponding label {-1, 1}. 1 stands for class A,
            -1 stands for class B.
    """
    feat = cal_feature(data)
    indices = (label==cls_A) + (label==cls_B)
    X, y = feat[indices], label[indices]
    ind_A, ind_B = y==cls_A, y==cls_B
    y[ind_A] = 1
    y[ind_B] = -1
    return X, y

def plot_feature(feature, y, plot_num, ax=None, classes=np.arange(10)):
    """plot the feature of different classes
    Input:
        feature: (n, 2), the feature matrix.
        y: (n, ) corresponding label.
        plot_num: int, number of samples for each class to be plotted.
        ax: matplotlib.axes.Axes, the axes to be plotted on.
        classes: array(0-9), classes to be plotted.
    Output:
        ax: matplotlib.axes.Axes, plotted axes.
    """
    cls_features = [feature[y==i] for i in classes]
    marks = ['s', 'o', 'D', 'v', 'p', 'h', '+', 'x', '<', '>']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple']
    if ax is None:
        _, ax = plt.subplots()
    for i, feat in zip(classes, cls_features):
        ax.scatter(*feat[:plot_num].T, marker=marks[i], color=colors[i], label=str(i))
    plt.legend(loc='upper right')
    plt.xlabel('intensity')
    plt.ylabel('asymmetry')
    return ax

def cal_error(theta, X, y, thres=1e-4):
    """calculate the binary error of the model w given data (X, y)
    theta: (d+1, 1), the weight vector
    X: (n, d), the data matrix [X, y]
    y: (n, ), the corresponding label
    """
    # Add a bias term to X
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    out = X_b @ theta - thres
    pred = np.sign(out)
    err = np.mean(pred.squeeze()!=y)
    return err

# prepare data
train_data, train_label = load_mat('p5/train_data.mat') # train_data: (7291, 16, 16), train_label: (7291, )
test_data, test_label = load_mat('p5/test_data.mat') # test_data: (2007, 16, 16), train_label: (2007, )

cls_A, cls_B = 1, 6
X, y, = cal_feature_cls(train_data, train_label, cls_A=cls_A, cls_B=cls_B)
X_test, y_test = cal_feature_cls(test_data, test_label, cls_A=cls_A, cls_B=cls_B)

# Add a bias term to the feature matrices
X_b = np.hstack([np.ones((X.shape[0], 1)), X])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# train
iters = 2000
d = 2
num_sample = X.shape[0]
threshold = 1e-4

# Perceptron and Pocket algorithm initialization
theta_p = np.zeros((d + 1, 1))
theta_pocket = np.zeros((d + 1, 1))
best_theta = np.zeros((d + 1, 1))
min_err = cal_error(best_theta, X, y)

# Lists to store errors
err_in_p = []
err_out_p = []
err_in_pocket = []
err_out_pocket = []

for iterate in range(iters):
    # Perceptron
    pred = np.sign(X_b @ theta_p)
    misclassified_indices = np.where(pred.squeeze() != y)[0]
    
    if len(misclassified_indices) > 0:
        # Pick a random misclassified point
        random_index = np.random.choice(misclassified_indices)
        xi = X_b[random_index, :].reshape(-1, 1)
        yi = y[random_index]
        theta_p = theta_p + yi * xi

    # Pocket
    pred_pocket = np.sign(X_b @ theta_pocket)
    misclassified_indices_pocket = np.where(pred_pocket.squeeze() != y)[0]

    if len(misclassified_indices_pocket) > 0:
        # Pick a random misclassified point
        random_index_pocket = np.random.choice(misclassified_indices_pocket)
        xi_pocket = X_b[random_index_pocket, :].reshape(-1, 1)
        yi_pocket = y[random_index_pocket]
        theta_pocket = theta_pocket + yi_pocket * xi_pocket
        
        # Check if the new theta is better
        current_err = cal_error(theta_pocket, X, y)
        if current_err < min_err:
            min_err = current_err
            best_theta = theta_pocket.copy()

    # Calculate and store errors
    err_in_p.append(cal_error(theta_p, X, y))
    err_out_p.append(cal_error(theta_p, X_test, y_test))
    err_in_pocket.append(cal_error(best_theta, X, y))
    err_out_pocket.append(cal_error(best_theta, X_test, y_test))


# plot Er_in and Er_out
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(iters), err_in_p, label='Perceptron In-sample Error')
plt.plot(range(iters), err_in_pocket, label='Pocket In-sample Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('In-sample Error vs. Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(iters), err_out_p, label='Perceptron Out-of-sample Error')
plt.plot(range(iters), err_out_pocket, label='Pocket Out-of-sample Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Out-of-sample Error vs. Iterations')
plt.legend()
plt.tight_layout()
plt.show()


# plot decision boundary
def plot_decision_boundary(X, y, theta, ax, title):
    # Plot data points
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], marker='+', label='1')
    ax.scatter(X[y==-1][:, 0], X[y==-1][:, 1], marker='*', label='6')
    
    # Plot decision boundary
    x1_min, x1_max = ax.get_xlim()
    x1 = np.array([x1_min, x1_max])
    
    # w0 + w1*x1 + w2*x2 = 0  => x2 = (-w0 - w1*x1) / w2
    w = theta.squeeze()
    if w[2] != 0:
        x2 = (-w[0] - w[1] * x1) / w[2]
        ax.plot(x1, x2, 'y-', label='Decision Boundary')
    
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Asymmetry')
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for 500 data points
plot_num = 500
indices_1 = np.where(y == 1)[0][:plot_num]
indices_6 = np.where(y == -1)[0][:plot_num]
plot_indices = np.concatenate([indices_1, indices_6])
X_plot, y_plot = X[plot_indices], y[plot_indices]

plot_decision_boundary(X_plot, y_plot, theta_p, axes[0], 'Perceptron Decision Boundary')
plot_decision_boundary(X_plot, y_plot, best_theta, axes[1], 'Pocket Decision Boundary')

plt.tight_layout()
plt.show()