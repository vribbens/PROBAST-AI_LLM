"""
1. Load consensus file (excel file with in row each question and columns for reviewers (LLM, Human and Consensus))
2. Calculate Cohen's kappa
3. Calculate ordinal agreement
4. Calculate consensus change resulting from LLM output
"""

# import modules and libraries
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import cohen_kappa_score

ORDINAL_SCALE = {"N": 0.0, "PN": 0.33, "PY": 0.67, "Y": 1.0}
LABELS = ['Y', 'PY', 'N', 'PN', 'NI', 'NA', 'high', 'low', 'unclear']

def Cohenskappa(df, col1, col2):
    LABELS = ['Y','PY','N','PN','NI','NA','high','low','unclear']
    sub = df[[col1, col2]].dropna().astype(str).apply(lambda s: s.str.strip())
    unknown = (set(sub[col1]) | set(sub[col2])) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    return cohen_kappa_score(sub[col1], sub[col2], labels=LABELS)

def disagreement_matrix(labels, power=1):
    """Pairwise disagreement in [0, 1]. power=1 linear, power=2 quadratic."""
    k = len(labels)
    w = np.zeros((k, k))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if a == b:
                w[i, j] = 0.0
            elif a in ORDINAL_SCALE and b in ORDINAL_SCALE:
                w[i, j] = abs(ORDINAL_SCALE[a] - ORDINAL_SCALE[b]) ** power
            else:
                w[i, j] = 1.0  # off-scale label involved: full disagreement
    return w

def ordinal_agreement(df, col1, col2, power=1):
    sub = df[[col1, col2]]

    # Check for unknown labels
    unknown = (set(sub[col1]) | set(sub[col2])) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    # observed confusion matrix (proportions)
    # a proportional 9x9 table where cell (i, j) counts the rows in which reviewer 1 said label i 
    # and reviewer 2 said label j. Each row of your dataset adds exactly one count to one cell.
    obs = pd.crosstab(sub[col1], sub[col2]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_numpy()
    obs = obs / obs.sum()

    # expected under independence, from marginals
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))

    w = disagreement_matrix(LABELS, power)
    observed_dis = (w * obs).sum()
    expected_dis = (w * exp).sum()

    return np.nan if expected_dis == 0 else 1 - observed_dis / expected_dis

def consensuschange(df, col1, col2, col3):
    sub = df[[col1, col2, col3]]
    disagreed = sub[sub[col1] != sub[col2]]                 # 1. rows where reviewer 1 and reviewer 2 disagreed
    changed = disagreed[disagreed[col3] != disagreed[col1]] # 2. of those, rows where reviewer 1's final answer differs from the original

    # 3. proportion relative to the disagreements
    if len(disagreed) == 0:
        return np.nan, 0, 0
    return len(changed) / len(disagreed), len(changed), len(disagreed)

def main(args):
    # load file
    excel = Path(args.excel)
    df = pd.read_excel(excel, sheet_name=None,  keep_default_na=False, na_values=[""]) # read all sheets
    df = pd.concat(df.values(), ignore_index=True)

    # Clear up columns
    df = df[[args.col1, args.col2, args.col3]].dropna().astype(str).apply(lambda s: s.str.strip()) # uniform formatting of columns
    df = df[~df[args.col3].eq('-')]     # drop rows with -
    print(f"Cohen's kappa: {round(Cohenskappa(df, args.col1, args.col2), 2)}")
    print(f"Weighted (linear) kappa:  {round(ordinal_agreement(df, args.col1, args.col2), 2)}")
    print(f"Weighted (quad.) kappa:   {round(ordinal_agreement(df, args.col1, args.col2, power=2), 2)}")

    rate, n_changed, n_disagreed = consensuschange(df, args.col1, args.col2, args.col3)
    print(f"Reviewer 1 changed {n_changed}/{n_disagreed} disagreed answers ({rate:.1%})")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel", 
                        default='', 
                        help="existing PROBAST workbook (not modified)")

    parser.add_argument("--col1", default="Human", help="first column for Cohen's kappa calculation")
    parser.add_argument("--col2", default="LLM", help="second column for Cohen's kappa calculation")
    parser.add_argument("--col3", default="Consensus", help="third column for consensus calculation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)