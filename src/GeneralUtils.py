import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mt


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
    




def silhouette_analysis(X, labels, ax, k):
    """_summary_

    Args:
        X (_type_): _description_
        labels (_type_): _description_
        ax (_type_): _description_
    """
    # clusters = [2, 3, 4 ,5, 6, 7]

    # performance
    #scores, labels = silhouettes(model, X, k, model_cat=model_type)
    scores = mt.silhouette_samples(X, labels)
    ss = mt.silhouette_score(X, labels)

    y_lower = 10
    plt.style.use('ggplot')
    for i in np.unique(labels):


        # Cluster scores
        ith_silhouette_values = scores[labels == i]
        ith_silhouette_values.sort()

        # size cluster
        size_cluster_i = len(ith_silhouette_values)
        y_upper = y_lower + size_cluster_i

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (k + 1)*10])
        
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouette_values )
        ax.set_title(f'N clusters = {k}')
        y_lower = y_upper + 10
        
    ax.axvline(x=ss, ymin=-0.1, ymax=1, ls='--', c='red', label='Average Silhouette Score')
    ax.legend(loc='upper right')