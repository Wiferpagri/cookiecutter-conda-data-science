import pandas as pd

def check_categories_proportion(
    df: pd.DataFrame, 
    col: str
    ) -> None:
    
    """
    Calculate and print the proportion of each category in a specified column of a DataFrame.

    This function takes a pandas DataFrame and a column name, calculates the proportion 
    (as a percentage) of each unique value (category) in the specified column, and prints 
    the resulting proportions.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    col (str): The name of the column for which to calculate category proportions.

    Returns:
    None: This function prints the category proportions and does not return any value.

    Example:
    >>> import pandas as pd
    >>> data = {'category': ['A', 'B', 'A', 'C', 'B', 'A']}
    >>> df = pd.DataFrame(data)
    >>> check_categories_proportion(df, 'category')
    A    50.0
    B    33.3
    C    16.7
    Name: category, dtype: float64
    """
    
    proportion = df[col].value_counts(normalize=True) * 100
    print(proportion)
