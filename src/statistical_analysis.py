import pandas as pd
from scipy import stats
import numpy as np

def desc_stats(df):
    num = df.select_dtypes(include=[np.number])
    return num.describe()

def corr(df):
    num = df.select_dtypes(include=[np.number])
    return num.corr()

def ci_mean(x, alpha=0.05):
    x = x.dropna()
    if len(x) < 2:
        return None
    mean = x.mean()
    sem = stats.sem(x)
    ci = stats.t.interval(1 - alpha, len(x) - 1, loc=mean, scale=sem)
    return mean, ci[0], ci[1], len(x)

def ttest_ind(g1, g2):
    g1 = g1.dropna()
    g2 = g2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return None
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return t, p, len(g1), len(g2)

def anova(groups_list):
    clean = [g.dropna() for g in groups_list if len(g.dropna()) > 0]
    if len(clean) < 2:
        return None
    f, p = stats.f_oneway(*clean)
    return f, p, [len(g) for g in clean]

def chi2(contingency_df):
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_df)
        return chi2, p, dof
    except Exception:
        return None
