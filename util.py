import pandas
import seaborn
from matplotlib import pyplot as plt

def generate_ordinal_encoding(column):
    """Map categorical features of a DataFrame column to int."""
    values = column.unique()
    mapping = {i: values[i] for i in range(len(values))}
    return mapping

def generate_ordinal_encoding_for_all_columns(dataset):
    """Generate ordinal encoding for all columns of a DataFrame."""
    encoding = {}
    for column in dataset.columns:
        if dataset[column].dtype == object:
            encoding[column] = generate_ordinal_encoding(dataset[column])
    return encoding

def apply_ordinal_encoding_to_all_columns(dataset, columns_dict):
    """Apply ordinal encoding on all columns of a DataFrame."""
    for column in columns_dict:
        dataset[column] = dataset[column].map({v: k for k, v in columns_dict[column].items()})

def apply_onehot_encoding(column):
    """Generate & apply onehot encoding (each value has boolean feature except NaN) of a DataFrame."""
    encoding = {}
    for feature in column.unique():
        # Check if the value is NaN (in Python, NaN != NaN)
        if feature == feature:
            encoding[feature] = column.apply(lambda x: x == feature)
    return pandas.DataFrame(encoding, index=column.index)

def draw_correlation_matrix(dataset, figsize=None, title="Correlation Heatmap"):
    """Draw correlation matrix."""
    if figsize:
        plt.figure(figsize=figsize)
    seaborn.heatmap(dataset.corr(), annot=True, fmt='.2f', linewidths=2, center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()