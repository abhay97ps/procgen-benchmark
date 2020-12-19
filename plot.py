import csv
import matplotlib.pyplot as plt
import numpy as np

def abcd(log1, log2):
    with open(log1, newline='') as readfile:
        reader = csv.reader(readfile)
        log1_data = list(reader)
    log1_data = log1_data[1:]
    log1_data = np.array(log1_data, dtype=np.float32)
    
    with open(log2, newline='') as readfile:
        reader = csv.reader(readfile)
        log2_data = list(reader)
    log2_data = log2_data[1:]
    log2_data = np.array(log2_data, dtype=np.float32)
    # todo check if timesteps have same range
    X = log1_data[:, 0]
    y1 = log1_data[:, 4]
    y2 = log2_data[:, 4]
    plt.plot(X, y1, '-b')
    plt.plot(X, y2, '-r')
    
    plt.legend(['IMPALA Baseline', 'Attention CNN'], loc='upper left')
    plt.show()