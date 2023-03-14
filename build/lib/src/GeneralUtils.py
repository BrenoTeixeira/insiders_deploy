import numpy as np

def sum_table(df_):

    """This function receives a dataset and returns a dataframe with information about each column of the dataset.
    Args:
        df_ (DataFrame): Dataset.

    Returns:
        DataFrame: returns a dataframe with the number of unique values and missing value of each column of a dataset.
    """

    summary = df_.dtypes.to_frame().rename(columns={0: 'dtypes'})
    summary['Uniques'] = df_.nunique()
    summary['Missing'] = df_.isnull().sum()
    summary['Missing %'] = np.round((df_.isnull().sum()/len(df_)).values*100, 2)
    summary = summary.reset_index().rename(columns={'index': 'Name'})
    return summary