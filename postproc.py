import pandas as pd
import numpy as np
import os


def fun():
    preds = ['8_con', '8_sin', '16_sin', '16_con', '16_con']
    preds = [pd.read_csv(os.path.join('predictions', 'prediction_' + p + '.csv')) for p in preds]
    solution = preds[-1].copy()
    for i in range(solution.shape[0]):
        values = [p.iloc[i, 1] for p in preds]
        values, counts = np.unique(values, return_counts=True)
        num_counts_max_idx = np.argmax(counts)
        num_counts_max = counts[num_counts_max_idx]
        if num_counts_max > len(preds) / 2:
            solution.iloc[i, 1] = values[num_counts_max_idx]
    solution.to_csv(os.path.join('predictions', 'garau.csv'), index=False)
    a=3


if __name__ == "__main__":
    fun()
