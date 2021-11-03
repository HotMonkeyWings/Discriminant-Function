import numpy as np
from math import pi, log

# Discriminant function as defined in the question
def discriminant_function(x, mean, cov, d, P):
    # Checking if the dimensions turn out to be scalars in the case only 1 feature is being taken.
    if d == 1:
        output = -0.5*(x - mean) * (1/cov)
        output = output * (x - mean)
        output += -0.5*d*log(2*pi) - 0.5*log(cov) 

    else: 
        output = np.matmul(-0.5*(x - mean), np.linalg.inv(cov))
        output = np.matmul(output, (x - mean).T)
        output += -0.5*d*log(2*pi) - 0.5*log(np.linalg.det(cov)) 
    
    # Adding Prior Probability
    output += (log(P) if P != 0 else 0)

    return output
        
def main():

    # Sample Data
    data = [
        # W1
        np.array([
            [-5.01, -8.12, -3.68],
            [-5.43, -3.48, -3.54],
            [1.08, -5.52, 1.66],
            [0.86, -3.78, -4.11],
            [-2.67, 0.63, 7.39],
            [4.94, 3.29, 2.08],
            [-2.51, 2.09, -2.59],
            [-2.25, -2.13, -6.94],
            [5.56, 2.86, -2.26],
            [1.03, -3.33, 4.33]
        ]),

        # W2
        np.array([
            [-0.91, -0.18, -0.05],
            [1.30, -2.06, -3.53],
            [-7.75, -4.54, -0.95],
            [-5.47, 0.50, 3.92],
            [6.14, 5.72, -4.85],
            [3.60, 1.26, 4.36],
            [5.37, -4.63, -3.65],
            [7.18, 1.46, -6.66],
            [-7.39, 1.17, 6.30],
            [-7.50, -6.32, -0.31]

        ]),

        # W3
        np.array([
            [5.35, 2.26, 8.13],
            [5.12, 3.22, -2.66],
            [-1.34, -5.31, -9.87],
            [4.48, 3.42, 5.19],
            [7.11, 2.39, 9.21],
            [7.17, 4.33, -0.98],
            [5.75, 3.97, 6.65],
            [0.77, 0.27, 2.41],
            [0.90, -0.43, -8.71],
            [3.52, -0.36, 6.43]
        ]) 
    ]

    # Measure mean and covariance
    means = []
    cov = []
    for i in range(len(data)):
        means.append(data[i].mean(axis=0))
        cov.append(np.cov(data[i].T))
    means, cov = np.array(means), np.array(cov)

    # General Configuration values
    n = len(data) - 1
    P = [0.5, 0.5, 0]
    g_values = [0 for i in range(n)]

    # Take input
    x = list(map(float, input("Enter the input vector: ").strip().split()))

    # Case A
    d = 1
    print("Case A: Using only feature vector x1")
    for i in range(n):
        g_values[i] = discriminant_function(x[0], means[i][0], cov[i][0][0], d, P[i])
    # Now to output the maximum result 
    result = g_values.index(max(g_values)) + 1
    print(x, "\twas classified as", result)
    
    # Case B
    d = 2
    print("\nCase B: Using only feature vectors x1 and x2")
    for i in range(n):
        g_values[i] = discriminant_function(x[0:2], means[i][0:2], cov[i][0:2, 0:2], d, P[i])
    # Now to output the maximum result 
    result = g_values.index(max(g_values)) + 1
    print(x, "\twas classified as", result)

    # Case C
    d = 3
    print("\nCase C: Using all feature vectors")
    for i in range(n):
        g_values[i] = discriminant_function(x, means[i], cov[i], d, P[i])
    # Now to output the maximum result 
    result = g_values.index(max(g_values)) + 1
    print(x, "\twas classified as", result)

if __name__ == '__main__':
    main()