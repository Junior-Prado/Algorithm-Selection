
import numpy as np
import pandas as pd
from s_dbw import SD, S_Dbw
from scipy.stats import spearmanr, skew, kurtosis, zscore, rankdata
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score


def metafeature_extraction(X, feature_type = "CaD"):
    # Function for extracting the meatafeatures of a dataset "X" (in array format). 
    # It has two modes for the two "feature_type" avalible, where:
    # 'CaD': are the metafeatures of pimentel 2019 'A new data characterization for selecting clustering algorithms using meta-learning'
    # 'distance': are the metafeatures of ferrari 2015 'Clustering algorithm selection by meta-learning systems: A new distance-based problem characterization and ranking combination method'
    # the function return a list of the 19 metafeatures of either work
    
    [n, p] = np.shape(X)  # dataset format
    size = int(n*(n-1)/2) # initial size

    if  feature_type == "CaD":
       
        R = rankdata(X, method = 'ordinal', axis=1)  # Ranking of instances  
        correlation =  np.zeros(size)                # Declaring empty vector
        dissimilarity =  np.zeros(size)              # Declaring empty vector
        size = 2*size - 1                            # ajusting the size for the combination of two vector
        Meta = np.zeros(size)                        # Declaring empty vector   
    
        # operation for calculating correlation and dissimilarity
        i=0
        for k in range(n):
            for l in range(k+1,n):              
                correlation[i] = spearmanr(R[k],R[l]).correlation
                dissimilarity[i] = np.linalg.norm(X[k] - X[l])
                i+=1    
        
        meta = np.concatenate([correlation,dissimilarity])  #concatenating the correlation and dissimilarity into a single vector
    
    elif feature_type == "distance":
        
        size = size-1          # ajusting size for [0,n-1] interval
        meta = np.zeros(size)  # Declaring empty vector
        Meta = np.zeros(size)  # Declaring empty vector
    
        # calculating distance
        i=0
        for k in range(n):
            for l in range(k+1,n-1):
                meta[i] = np.linalg.norm(X[k] - X[l])
                i+=1
    
    # normalizing the metadata vector
    for i in range(size):
        Meta[i] = (meta[i] - np.min(meta)) / (np.max(meta) - np.min(meta))

    zs_meta = np.abs(zscore(Meta))    # calculating absolute of zscore from metadata array
    MF = np.zeros(19)                 # empty vector for storing the metafeatures
    
    MF[0] = np.mean(Meta)
    MF[1] = np.var(Meta)
    MF[2] = np.std(Meta)
    MF[3] = skew(Meta)
    MF[4] = kurtosis(Meta)
    
    MF[5]  = np.count_nonzero((Meta >= 0.0) & (Meta <  0.1))/size
    MF[6]  = np.count_nonzero((Meta >= 0.1) & (Meta <  0.2))/size
    MF[7]  = np.count_nonzero((Meta >= 0.2) & (Meta <  0.3))/size
    MF[8]  = np.count_nonzero((Meta >= 0.3) & (Meta <  0.4))/size
    MF[9]  = np.count_nonzero((Meta >= 0.4) & (Meta <  0.5))/size
    MF[10] = np.count_nonzero((Meta >= 0.5) & (Meta <  0.6))/size
    MF[11] = np.count_nonzero((Meta >= 0.6) & (Meta <  0.7))/size
    MF[12] = np.count_nonzero((Meta >= 0.7) & (Meta <  0.8))/size
    MF[13] = np.count_nonzero((Meta >= 0.8) & (Meta <  0.9))/size
    MF[14] = np.count_nonzero((Meta >= 0.9) & (Meta <= 1.0))/size
    
    MF[15] = np.count_nonzero((zs_meta >= 0.0) & (zs_meta < 1.0))/size
    MF[16] = np.count_nonzero((zs_meta >= 1.0) & (zs_meta < 2.0))/size
    MF[17] = np.count_nonzero((zs_meta >= 2.0) & (zs_meta < 3.0))/size
    MF[18] = np.count_nonzero((zs_meta >= 3.0))/size
    
    return MF



def performance_extraction(X, n_clusters=2, scoring_metric=None, ranking_method = 'min'):
    
    # Function that takes a dataset "X" and performs various clustering algorithms,
    # Followed by an evaluation of each by a chosen "scoring_metric" and then this score is ranked,
    # this ranking, called here the performance, is saved to a dataframe and return.
    #
    # This function takes the following parameters:
    # X (array) -> parameter with the dataset.
    # n_clusters (int) -> parameter with number of clusters, default to 2.
    # scoring_metric (function) -> parameter the internal validation index, if none is chosen default to silhouette.
    # ranking_method (strin) -> parameter that asks the objective of the scoring_metric where:
    #    min (default): the lowest value is placed first
    #    max: the highest value is placed first

    
    if scoring_metric == None:
        scoring_metric = silhouette_score
    
    if ranking_method != 'min' and ranking_method != 'max':
        raise ValueError("Error: ranking_method must be 'min' or 'max'") 
    
    # Running all clustering algoritms and saving the resulting labels on a dictionary
    clusters_dict = {
        'Ward_Agglomerative_Clustering':     AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward').fit(X).labels_,
        'Complete_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = n_clusters, linkage = 'complete').fit(X).labels_,
        'Average_Agglomerative_Clustering':  AgglomerativeClustering(n_clusters = n_clusters, linkage = 'average').fit(X).labels_,
        'Single_Agglomerative_Clustering':   AgglomerativeClustering(n_clusters = n_clusters, linkage = 'single').fit(X).labels_,
        'K_Means':                           KMeans(n_clusters = n_clusters).fit(X).labels_,
        'Mini_Batch_K_Means':                MiniBatchKMeans(n_clusters = n_clusters).fit(X).labels_,
        'Spectral_Clustering':               SpectralClustering(n_clusters = n_clusters).fit(X).labels_,
        'Affinity_Propagation':              AffinityPropagation().fit(X).labels_,
        'Gaussian_Mixture':                  GaussianMixture(n_components=n_clusters).fit_predict(X)
        }
    
    algorithms_names = list(clusters_dict.keys()) # Extracting algorithms_names from dictionary and making empty score list

    scores = []
    for key, labels in clusters_dict.items():
        scores.append(scoring_metric(X, labels))

    ranked_scores = rankdata(scores, method = 'ordinal')
    if ranking_method == 'max':
        ranked_scores = len(clusters_dict)+1-ranked_scores
        
    performance = pd.DataFrame({
        'Name':algorithms_names,
        'Rank':ranked_scores,
        'Score':scores
        })

    return performance



def metadata_extraction(database, n_clusters=2, feature_type = "CaD"):
    # Function that takes extracts both the performance and the metafeatures for a entire database in a range of number of clusters
    #
    # This function takes the following parameters:
    # database (list) -> list of datasets (in dataframe format).
    # n_clusters (int) -> number of clursters
    # feature_type (string) -> specifies the type of metafeature ('CaD' or 'distance')
    #
    # this function returns a tuple with two dataframes, one with the metafeatures and other with the performance 
    
    performances_columns = list(performance_extraction([[2],[1],[-1],[-2]], n_clusters = 2)['Name'])
    metafeatures_columns = [f'MF{i}' for i in range(1,20)]
    
    performances = pd.DataFrame(columns = performances_columns)
    metafeatures = pd.DataFrame(columns = metafeatures_columns)
 
    for dataset in database:
                                                             
        X = dataset.to_numpy()
        MetaFeatures = list(metafeature_extraction(X, feature_type = feature_type))

        # calculating performance of the clusterring methods using the chosen validetion indexes
        algorithm_ranking_dict = {
            'Calinski_Harabasz': performance_extraction(X, n_clusters, scoring_metric = calinski_harabasz_score, ranking_method = 'max')['Rank'].values,
            'Silhouette':        performance_extraction(X, n_clusters, scoring_metric = silhouette_score,        ranking_method = 'max')['Rank'].values,
            'Davies_Bouldin':    performance_extraction(X, n_clusters, scoring_metric = davies_bouldin_score,    ranking_method = 'min')['Rank'].values,
            'SD':                performance_extraction(X, n_clusters, scoring_metric = SD,                      ranking_method = 'min')['Rank'].values,
            'S_Dbw':             performance_extraction(X, n_clusters, scoring_metric = S_Dbw,                   ranking_method = 'min')['Rank'].values
            }

        avg_rank = sum(algorithm_ranking_dict.values())/len(algorithm_ranking_dict)    # calculating the average rank of all ranking methods
        rank = list(rankdata(avg_rank, method = 'ordinal'))                            # re-assigning the ranks by 
        metafeatures.loc[len(metafeatures)] = MetaFeatures                             # adding the new metafeatures to the metafeatures database
        performances.loc[len(performances)] = rank                                     # adding the new performance to the performance database
        
    return (metafeatures,performances)



def batch_metadata_extraction(database, n_clusters_range=range(2,11), feature_type = "CaD"):
    # Function that takes extracts both the performance and the metafeatures for a entire database in a range of number of clusters
    #
    # This function takes the following parameters:
    # database (list) -> list of datasets (in dataframe format).
    # n_clusters_range (list) -> range of integers that will be
    # feature_type (string) -> specifies the type of metafeature ('CaD' or 'distance')
    #
    # this function return a single dataframe that combines both performance and metafeatures for multiples numbers of clusters,
    # note that metafeatures are the same independent of the number of clursters, but the performance changes,
    # so metadata only makes sense for one given number of clusters, ant to use the data of this function you will need to filter it
    # the filter_data function will do it for you
    
    dataset_index = 0
    algorithms_names = algorithms_names = ['Ward_Agglomerative_Clustering', 'Complete_Agglomerative_Clustering', 'Average_Agglomerative_Clustering', 'Single_Agglomerative_Clustering', 'K_Means', 'Mini_Batch_K_Means', 'Spectral_Clustering', 'Affinity_Propagation', 'Gaussian_Mixture']
    metafeatures_names = [f'MF{i}' for i in range(1,20)]
    columns = ['dataset_index','n_clusters'] + metafeatures_names + algorithms_names
    metabase = pd.DataFrame(columns = columns)

    for dataset in database:
                                                             
        X = dataset.to_numpy()
        MetaFeatures = list(metafeature_extraction(X, feature_type = feature_type)) # calculating the metafeatures and converting results to list,

        for n_clusters in n_clusters_range:

            # calculating performance of the clusterring methods using the chosen validetion indexes
            algorithm_ranking_dict = {
                'Calinski_Harabasz': performance_extraction(X, n_clusters, scoring_metric = calinski_harabasz_score, ranking_method = 'max')['Rank'].values,
                'Silhouette':        performance_extraction(X, n_clusters, scoring_metric = silhouette_score,        ranking_method = 'max')['Rank'].values,
                'Davies_Bouldin':    performance_extraction(X, n_clusters, scoring_metric = davies_bouldin_score,    ranking_method = 'min')['Rank'].values,
                'SD':                performance_extraction(X, n_clusters, scoring_metric = SD,                      ranking_method = 'min')['Rank'].values,
                'S_Dbw':             performance_extraction(X, n_clusters, scoring_metric = S_Dbw,                   ranking_method = 'min')['Rank'].values
                }

            avg_rank = sum(algorithm_ranking_dict.values())/len(algorithm_ranking_dict)    # calculating the average rank of all ranking methods
            rank = list(rankdata(avg_rank, method = 'ordinal'))                            # re-assigning the ranks by 
            metabase.loc[len(metabase)] = [dataset_index,n_clusters] + MetaFeatures + rank # adding the new metadata to the metadata database
            print(f'i={dataset_index}, k={n_clusters}')
    
        dataset_index+=1

    metabase['n_clusters'] = metabase['n_clusters'].astype(int) 
    metabase['dataset_index'] = metabase['dataset_index'].astype(int)
    return metabase




