import pandas as pd
import numpy as np

def eda_summary(df):
    info = {
        "shape": df.shape,
        "cols": list(df.columns),
        "nulls": df.isnull().sum().to_dict(),
        "dupes": df.duplicated().sum()
    }
    return info

def iqr_outliers(s):
    s = s.dropna()
    if len(s) == 0:
        return 0
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((s < low) | (s > high)).sum()

def z_outliers(s, thresh=3):
    s = s.dropna()
    if len(s) == 0:
        return 0
    std = s.std(ddof=0)
    if std == 0:
        return 0
    z = (s - s.mean()) / std
    return (np.abs(z) > thresh).sum()
