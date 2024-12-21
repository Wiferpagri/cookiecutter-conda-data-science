import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_numerical_dist(
    df: pd.DataFrame, 
    cols: list, 
    hue=None, 
    name_df='Train data'
    ) -> None:
    """
    Visualize the distribution of numerical columns in a DataFrame using histograms and boxplots.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    cols (list): List of numerical columns to visualize.
    hue (str, optional): Column name to use for color coding in the plots. Defaults to None.
    df_name (str, optional): Name of the DataFrame to include in plot titles. Defaults to 'Train data'.

    Returns:
    None: The function creates and displays the plots.

    """
    
    fig, axs = plt.subplots(len(cols), 2, figsize=(12, 6*len(cols)))

    if hue:
        for i, col in enumerate(cols):
            sns.histplot(data=df, x=col, hue=hue, ax=axs[i, 0], kde=True)
            sns.boxplot(data=df, x=col, hue=hue, ax=axs[i, 1])

            axs[i, 0].set_title(f'{name_df} {col} histogram by {hue}')
            axs[i, 1].set_title(f'{name_df} {col} boxplot by {hue}')

    else:
        for i, col in enumerate(cols):
            sns.histplot(data=df, x=col, ax=axs[i, 0], kde=True)
            sns.boxplot(data=df, x=col, ax=axs[i, 1])

            axs[i, 0].set_title(f'{name_df} {col} histogram')
            axs[i, 1].set_title(f'{name_df} {col} boxplot')
    
    plt.tight_layout()


def plotting_categories(
    df: pd.DataFrame, 
    cat_cols: list, 
    hue=None, 
    name_df = 'Train data'
    ) -> None:
    """
    Plot count and proportion distribution of categorical columns in a DataFrame.

    This function generates subplots for each categorical column specified in `cat_cols`.
    Depending on the presence of `hue`, it either plots two subplots (countplot and proportion histogram)
    for each column when `hue` is specified, or just a countplot when `hue` is not specified.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    cat_cols (list): List of column names (strings) containing categorical data to be plotted.
    hue (str, optional): Optional categorical variable for grouping/counting.
    name_df (str, optional): Name of the DataFrame for labeling plots (default is 'Train data').

    Returns:
    None: This function directly plots the graphs using Matplotlib and Seaborn.

    Example:
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = {'category': ['A', 'B', 'A', 'C', 'B', 'A'], 'value': [1, 2, 3, 1, 2, 3]}
    >>> df = pd.DataFrame(data)
    >>> plotting_categories(df, ['category'])
    """
    if hue:
        fig, axs = plt.subplots(len(cat_cols), 2, figsize=(12, 6*len(cat_cols)))
        for i, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, hue=hue, ax=axs[i, 0])
            sns.histplot(data=df, x=col, hue=hue, multiple='fill', stat='proportion', shrink=.8, ax=axs[i, 1])

            axs[i, 0].set_title(f'{name_df} {col} count by {hue}')
            axs[i, 1].set_title(f'{name_df} {col} proportion by {hue}')


    else:
        fig, axs = plt.subplots(len(cat_cols), 1, figsize=(6, 6*len(cat_cols)))
        for i, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, ax=axs[i])
            axs[i].set_title(f'{name_df} {col} count')
    
    plt.tight_layout()
