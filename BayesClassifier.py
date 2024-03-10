import numpy as np
import pdb

def pdf(x, mean, sigma):
    # Define the mean and standard deviation
    mean = mean
    std_dev = sigma
    # Calculate the Gaussian PDF
    pdf = np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / (std_dev * np.sqrt(2 * np.pi))

    return pdf

def bayes_pos(p_maj, x1, x2):
    return (p_maj * pdf(x1, 1, 0.5) * pdf(x2, 1, 0.5) + (1-p_maj) * pdf(x1, 1, 0.5) * pdf(x2, -1, 0.5))

def bayes_neg(p_maj, x1, x2):
    return (p_maj * pdf(x1, -1, 0.5) * pdf(x2, -1, 0.5) + (1-p_maj) * pdf(x1, -1, 0.5) * pdf(x2, 1, 0.5))

def clf(x1, x2):
    p_pos = bayes_pos(0.9, x1, x2)
    p_neg = bayes_neg(0.9, x1, x2)
    print('p_pos', p_pos)
    print('p_neg', p_neg)
    if p_pos > p_neg:
        return 1
    elif p_pos < p_neg:
        return -1
    else:
        return 0

x1 = 0.2
x2 = -1
print(clf(x1, x2))
pdb.set_trace()
