import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mt
import sklearn.mixture as mix
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd


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

    """This function returns the segment of a customer based o his RFM score.

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
    """This function receives a Dataframe or data-matrix, the label (cluster) of each instance in the dataframe, the plot axis and the number of clusters. And plots the silhoutte analysis plot.

    Args:
        X (array-like or dataframe): Input data matrix or Pandas DataFrame.
        labels (array): Array with the assigned cluster of each instance in the dataset.
        ax (axis): Axis to plot the silhoutte analysis graphic.
    """

    # performance
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


def kmeans_performance(X, clusters, title='', metric='euclidean', plot=True):


    """
    Plot the performance of KMeans algorithm on a dataset as a function of the number of clusters.
    
    Parameters:
        X (array-like or dataframe): Input data matrix or Pandas DataFrame.
        title (str): Title for the plot.
        clusters (list or array-like): List of integers representing the number of clusters to evaluate.
        metric (str, optional): Distance metric to use for computing cluster similarity. Default is 'euclidean'.
        plot (boolean, optional): If True, plots the performance graphic. If False, returns the metrics in a table format.
    Returns:
        None (displays a plot)
    """

    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("Input data must be a numpy array or a pandas dataframe.")
    if not isinstance(title, str):
        raise ValueError('Title must be a string')
    if not isinstance(clusters, (list, np.ndarray)):
        raise ValueError('Clusters must be a list or numpy array of integer values')
    
    if not isinstance(metric, str):
        raise ValueError('metric must be a string')
    
    
    if not metric in metrics:
        raise ValueError(f"Unknown distance: {metric}\n Possible Values: {metrics}")
    kmeans_list = []
    for k in clusters:
        
        kmeans_model = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300, random_state=42 )

        kmeans_model.fit(X)

        labels = kmeans_model.predict(X)

        # performance
        sil = mt.silhouette_score(X, labels, metric=metric)

        kmeans_list.append(sil)

    if plot:
        plt.plot(clusters, kmeans_list)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title(title + ' Elbow Plot')
        return None
    
    else:
        df_results = pd.DataFrame({'KMeans': kmeans_list}).T

    return df_results


def gmm_performance(X, components,  title='', metric='euclidean', covariance_type='full', plot=True):


    """
    Plot the performance of KMeans algorithm on a dataset as a function of the number of clusters.
    
    Parameters:
        X (array-like or dataframe): Input data matrix or Pandas DataFrame.
        title (str): Title for the plot.
        clusters (list or array-like): List of integers representing the number of clusters to evaluate.
        metric (str, optional): Distance metric to use for computing cluster similarity. Default is 'euclidean'.
        covariance_type (str, optional): Covariance Type. Default is 'full'.
    Returns:
        None (displays a plot)
    """

    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("Input data must be a numpy array or a pandas dataframe.")
    if not isinstance(title, str):
        raise ValueError('Title must be a string')
    if not isinstance(components, (list, np.ndarray)):
        raise ValueError('Clusters must be a list or numpy array of integer values')
    
    if not isinstance(metric, str):
        raise ValueError('metric must be a string')
    
    if not metric in metrics:
        raise ValueError(f"Unknown distance: {metric}\n Possible Values: {metrics}")
    
    if not isinstance(metric, str):
        raise ValueError('covariance must be a string')
    
    gmm_list = []
    
    for k in components:
        gmm = mix.GaussianMixture(n_components=k, random_state=42, covariance_type=covariance_type, n_init=300)

        try:
            gmm.fit(X)
            labels = gmm.predict(X)

            ss = mt.silhouette_score(X, labels)
        except FloatingPointError:
            ss = np.NaN
        
        gmm_list.append(ss)

    if plot:
        plt.plot(components, gmm_list)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title(title + ' Elbow Plot')
        return None
    
    else:

        df_results = pd.DataFrame({'Gaussian Mixture': gmm_list} ).T

        return df_results


def hierarchical_performance(X, clusters, title='', metric='euclidean', plot=True):
    
    """
    Plot the performance of KMeans algorithm on a dataset as a function of the number of clusters.
    
    Parameters:
        X (array-like or dataframe): Input data matrix or Pandas DataFrame.
        title (str): Title for the plot.
        clusters (list or array-like): List of integers representing the number of clusters to evaluate.
        metric (str, optional): Distance metric to use for computing cluster similarity. Default is 'euclidean'.
        
    Returns:
        None (displays a plot)
    """
        

    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("Input data must be a numpy array or a pandas dataframe.")
    if not isinstance(title, str):
        raise ValueError('Title must be a string')
    if not isinstance(clusters, (list, np.ndarray)):
        raise ValueError('Clusters must be a list or numpy array of integer values')
    
    if not isinstance(metric, str):
        raise ValueError('metric must be a string')
    
    
    if not metric in metrics:
        raise ValueError(f"Unknown distance: {metric}\n Possible Values: {metrics}")
    

    agg_list = []
    for k in clusters:
        
        agg_clu = AgglomerativeClustering(n_clusters=k)

        labels = agg_clu.fit_predict(X)

        sil = mt.silhouette_score(X, labels, metric=metric)

    

        agg_list.append(sil)

    if plot: 

        plt.plot(clusters, agg_list)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title(title + ' Elbow Plot')

        return None
    
    else:

        df_results = pd.DataFrame({'H_cluster': agg_list} ).T

        return df_results
