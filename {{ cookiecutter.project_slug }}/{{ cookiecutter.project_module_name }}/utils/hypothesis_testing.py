import numpy as np
import pandas as pd
import pingouin

def chi2_independence_test(
    data: pd.DataFrame, 
    main_cat: str, 
    sec_cat: str, 
    sig_col: list = [], 
    no_sig_col: list = [], 
    alpha=0.05
    ) -> list:
    """
    Perform a chi-squared test of independence between two categorical variables.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data.
    main_cat : str
        The name of the main categorical variable (column) in the DataFrame.
    sec_cat : str
        The name of the secondary categorical variable (column) in the DataFrame.
    sig_col : list
        A list to append the name of the secondary categorical variable if the test is significant.
    no_sig_col : list
        A list to append the name of the secondary categorical variable if the test is not significant.
    alpha : float, optional (default=0.05)
        The significance level for the test.

    Returns:
    --------
    sig_col : list
    no_sig_col : list

    Prints:
    -------
    - The chi-squared test statistics.
    - A message indicating whether the null hypothesis is rejected or not.
    - Appends the secondary categorical variable name to `sig_col` if the null hypothesis is rejected.
    - Appends the secondary categorical variable name to `no_sig_col` if the null hypothesis is not rejected.

    Notes:
    ------
    - The function uses the `pingouin` library to perform the chi-squared test of independence.
    - The null hypothesis is that the two categorical variables are independent.
    - If the p-value is less than the significance level (`alpha`), the null hypothesis is rejected, indicating a statistically significant difference between the categories.
    """
    
    expected, observed, stats = pingouin.chi2_independence(data=data, x=main_cat, y=sec_cat)
    
    stats = pd.DataFrame(stats)
    
    print('-'*80)
    print(stats)
        
    stat_sign = stats['pval'] < alpha
    
    if stat_sign.any():
        print(f'Reject null hypothesis: There is a statistically significant difference between {sec_cat} and {main_cat}')
        sig_col.append(sec_cat)
    else:
        print(f'Failed to reject null hypothesis: There is no statistically significant difference {sec_cat} and {main_cat}')
        no_sig_col.append(sec_cat)
    print('-'*80)
    
    return sig_col, no_sig_col

def kruskal_wallis_test(
    data: pd.DataFrame, 
    cat_feat: str, 
    num_feat: str, 
    sig_col: list = [], 
    no_sig_col: list = [], 
    alpha=0.05
    ) -> list:
    """
    Perform a Kruskal-Wallis test to determine if there are statistically significant differences in a numerical feature across categories of a categorical feature.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data.
    cat_feat : str
        The name of the categorical feature (column) in the DataFrame.
    num_feat : str
        The name of the numerical feature (column) in the DataFrame.
    sig_col : list
        A list to append the name of the categorical feature if the test is significant.
    no_sig_col : list
        A list to append the name of the categorical feature if the test is not significant.
    alpha : float, optional (default=0.05)
        The significance level for the test.

    Returns:
    --------
    sig_col : list
    no_sig_col : list

    Prints:
    -------
    - The Kruskal-Wallis test results.
    - A message indicating whether the null hypothesis is rejected or not.
    - Appends the categorical feature name to `sig_col` if the null hypothesis is rejected.
    - Appends the categorical feature name to `no_sig_col` if the null hypothesis is not rejected.

    Notes:
    ------
    - The function uses the `pingouin` library to perform the Kruskal-Wallis test.
    - The null hypothesis is that the distributions of the numerical feature are the same across the categories of the categorical feature.
    - If the p-value is less than the significance level (`alpha`), the null hypothesis is rejected, indicating a statistically significant difference in the numerical feature between the categories of the categorical feature.
    """
    
    results = pingouin.kruskal(data=data, dv=num_feat, between=cat_feat)
    
    print('-'*80)
    print(results)
    
    if results['p-unc'].iloc[0] < alpha:
        print(f'Reject null hypothesis: There is a statistically significant difference in {num_feat} between {cat_feat}.')
        sig_col.append(num_feat)
    else:
        print(f'Failed to reject null hypothesis: There is no statistically significant difference in {num_feat} between {cat_feat}')
        no_sig_col.append(num_feat)
    print('-'*80)
    
    return sig_col, no_sig_col
