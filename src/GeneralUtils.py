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


def segment(score):
    """This function returns the segment a a customer based o his RFM score.

    """


    if score in [555, 554, 544, 545, 454, 455, 445]:
        return 'Champions'
     
    if score in [543, 444, 435, 355, 354, 345, 344, 335]: 
        return 'Loyal'
    if score in [553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323]: 
        return 'Potential Loyalist'

    if score in [512, 511, 422, 421, 412, 411, 311]: 
        return 'New Customers'

    if score in [525, 524, 523, 522, 521, 515, 514, 513, 425,424, 413,414,415, 315, 314, 313]: 
        return 'Promising'

    if score in [535, 534, 443, 434, 343, 334, 325, 324]: 
        return 'Need Attention'

    if score in [331, 321, 312, 221, 213, 231, 241, 251]: 
        return 'About To Sleep'

    if score in [255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124]: 
        return 'At Risk'

    if score in [155, 154, 144, 214,215,115, 114, 113]: 
        return 'Cannot Lose Them'

    if score in [332, 322, 233, 232, 223, 222, 132, 123, 122, 212, 211]: 
        return 'Hibernating customers'

    if score in [111, 112, 121, 131,141,151]: 
        return 'Lost customers'