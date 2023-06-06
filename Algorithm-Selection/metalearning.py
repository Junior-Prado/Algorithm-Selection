import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import spearmanr, rankdata
from metadata_derivation import metafeature_extraction


def filter_data(metadata, n_clusters = 2):
    # Function that process data from batch_metadata_extraction to get it ready for metalearning, 
    # filtering the batch data to a single n_clusters and separating metafeature data from performance data
    #
    # This function takes the following parameters:
    # metadata (dataframe) -> result of applying batch_metadata_extraction to a database, contains a lot of data packed in
    # n_clusters (int) -> the number of clusters to filter the data by.
    #
    # this function returns a tuple with two dataframes, one with the metafeatures and other with the performance 
    
    metadata = metadata[metadata['n_clusters'] == n_clusters] # filtering data with chosen number of clusters
    metadata.drop(columns=['n_clusters','dataset_index'])     # removing now unneeded columns
    metadata.reset_index()                                    # reorganizing index
    metafeatures = metadata.iloc[:,2:21]                      # separate metafeature data
    performances = metadata.iloc[:,21:29]                     # separate performance data

    return (metafeatures, performances)



def load_test_metadata(n_clusters = 2, feature_type = "CaD"):
    # Function that loads premade data for a meta learning system
    #
    # This function takes the following parameters:
    # n_clusters (int) -> the number of clusters to filter the data by.
    # feature_type (string) -> specifies the type of metafeature ('CaD' or 'distance')
    #
    # this function returns a tuple with two dataframes, one with the metafeatures and other with the performance 

    if feature_type == "CaD":
        metadata = pd.read_csv('metadata/CaD.csv')
    elif feature_type == "distance":
        metadata = pd.read_csv('metadata/distance.csv') 
    else:
        raise NameError("Invalid feature_type, must be 'CaD' or 'distance'")

    return filter_data(metadata, n_clusters)



def predict_performance(X, metadata = None, n_neighbors = 20, feature_type = "CaD"):
    
    # if no metadata is provided, load a pre-made metadata
    if metadata == None:
        (metafeatures, performances) = load_test_metadata(n_clusters = 2, feature_type = feature_type)
    else:
        (metafeatures, performances) = metadata
    
    
    # if X has the same shape as a metafeature vector, skip the metafeature extraction step, as the user already feed the function with metafeatures
    if X.shape != (1,19):
        MF = [metafeature_extraction(X, feature_type = feature_type)]
    else:
        MF=X

    distance, index = KDTree(metafeatures).query(MF, k=n_neighbors) # Building and Quering a KDtree to find the index of the datasets that have metafeatures closest to those of X

    # Making, filling and standardizing a weights vector where:
    # the weight is 1 if the index of the vectors corresponds to de index of one of its nearest neighbours
    # the weight is 0 if the index of the vectors corresponds to de index of one of its nearest neighbours
    [n, m] = metafeatures.shape
    weights = np.zeros(n)
    j = 0
    for i in index:
        weights[i] = 1/(distance[j]+0.000001)
    weights = weights/weights.sum()
    
    rank_score = weights.dot(performances)          #
    rank = rankdata(rank_score, method = 'ordinal') #

    predicted_performance = pd.DataFrame({
        'Name': performances.columns,
        'Rank': rank,
        })

    return predicted_performance



def get_all_scores(data):

    min_k = int(data['n_clusters'].min())
    max_k = int(data['n_clusters'].max()+1)
    min_i = int(data['dataset_index'].min())
    max_i = int(data['dataset_index'].max()+1)  
    
    columns = ['n_clusters', 'n_neighbors', 'dataset_index', 'correlation']
    all_scores = pd.DataFrame(columns = columns)
    

    for n_clusters in range(min_k, max_k):
        for n_neighbors in range(1, max_i):
            for dataset_index in range(min_i, max_i):
                print(f'dataset_index = {dataset_index} - n_neighbors = {n_neighbors} - n_clusters = {n_clusters}')
                
                test = data.loc[(data['dataset_index'] == dataset_index)]           #
                train = data.loc[(data['dataset_index'] != dataset_index)]          #
                (MF, real_performance) = filter_data(test, n_clusters = n_clusters) #
                metadata = filter_data(train, n_clusters = n_clusters)              #
                
                predicted_performance = predict_performance(MF, metadata, n_neighbors)       #
                predicted_performance = list(predicted_performance['Rank'])                  #
                real_performance = list(real_performance.iloc[0])                            #     
                correlation = spearmanr(real_performance, predicted_performance).correlation #
            
                all_scores.loc[len(all_scores)] = [n_clusters, n_neighbors, dataset_index, correlation]
    
    return all_scores
