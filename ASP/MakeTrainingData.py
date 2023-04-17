def Feature_Extractor(X, feature_type = "CaD"):
    
    import numpy as np
    from scipy.stats import spearmanr, skew, kurtosis, zscore, rankdata
    
    # dataset format
    [n, p] = np.shape(X)
    size = int(n*(n-1)/2)

    if  feature_type == "CaD":
        # Ranking of instances
        R = rankdata(X, method = 'min', axis=1)
    
        # Declaring empty vectors
        correlation =  np.zeros(size)
        dissimilarity =  np.zeros(size)
        
        # ajusting the size for the combined vector
        size = 2*size-1 
        Meta = np.zeros(size)
    
        # calculating correlation and dissimilarity
        i=0
        for k in range(n):
            for l in range(k+1,n):
                correlation[i] = spearmanr(R[k],R[l]).correlation
                dissimilarity[i] = np.linalg.norm(X[k] - X[l])
                i+=1
        
        # concatenating the correlation and dissimilarity into a metadata vector
        meta = np.concatenate([correlation,dissimilarity])
    
    elif feature_type == "distance":
        
        # ajusting size for [0,n-1] interval
        size = size-1 
        meta = np.zeros(size)
        Meta = np.zeros(size)
    
        # calculating  dissimilarity
        i=0
        for k in range(n):
            for l in range(k+1,n-1):
                meta[i] = np.linalg.norm(X[k] - X[l])
                i+=1
    
    # normalizing the metadata vector
    m_min = np.min(meta)
    m_max = np.max(meta)
    for i in range(size):
        Meta[i] = (meta[i] - m_min) / (m_max - m_min)
        
    # calculating zscore from metadata array
    zs_meta = np.abs(zscore(Meta))

    # calculating and storing metafeatures
    MF = np.zeros(19)
    
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

def Clustering_Evaluation(X, k=2, metric=None, method='max', random_state = None):
    
    # Function that takes a parameters of varios clustering algoritms and returns the performace ranked
    
    # 'X' is the dataset
    # 'k' is an integer describing the number of clusters
    # 'metric' is an function describing the scoring method, the metric must take a dataset and a labels only 
    # 'method' is a stirng that decides the ranking method, where:
    #     'max' means higher score
    #     'min' means minimun score
    #     'absolute' means score closer to zero
    # 'random_state' is the seed for algoritms that need it

        
    # Importing functions
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, SpectralClustering
        
    if random_state == None:
        from random import randint
        random_state == randint(0,1000)
    
    # Running clustering algoritms and saving objects on a dictionary
    clusters_dict = {
        'Ward_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = k, linkage = 'ward').fit(X).labels_,
        'Complete_Agglomerative_Clustering':AgglomerativeClustering(n_clusters = k, linkage = 'complete').fit(X).labels_,
        'Average_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = k, linkage = 'average').fit(X).labels_,
        'Single_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = k, linkage = 'single').fit(X).labels_,
        'K_Means':KMeans(n_clusters = k, random_state = random_state).fit(X).labels_,
        'Mini_Batch_K_Means': MiniBatchKMeans(n_clusters = k, random_state = random_state).fit(X).labels_,
        'Spectral_Clustering': SpectralClustering(n_clusters = k).fit(X).labels_
        }
    
    # Extracting algorithms_names from dictionary and making empty score list
    algorithms_names = list(clusters_dict.keys())

    # scoring each clustering method acording to chosen metric function
    # if none default to silhouette score   
    scores = []
    if metric == None:
        from sklearn.metrics import silhouette_score
        for key, labels in clusters_dict.items():
            scores.append(silhouette_score(X, labels))
    else:
        for key, labels in clusters_dict.items():
            scores.append(metric(X, labels))
    
    # deciding the ranking method
    if (method == 'min') or (method == 'max'):
        ranked_scores = rankdata(scores, method = 'ordinal')
        
        # if the method is max just flip the rank of the min method
        if method == 'max':
            ranked_scores = len(clusters_dict)+1-ranked_scores  
    
    # if method is absolute take the absolute of the scores then repeat step of min
    elif method == 'absolute':
        abs_scores = np.absolute(scores)
        ranked_scores = rankdata(abs_scores, method = 'ordinal')
    
    else:
        raise ValueError("Error: 'method' must be 'min', 'max' or 'absolute") 
        
    # puting data in a dataframe dataframe
    df = pd.DataFrame({
        'Name':algorithms_names,
        'Rank':ranked_scores,
        'Score':scores
        })

    return df

def metadata_extractor(database, k_range=[2], feature_type = "CaD"):
    from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
    from scipy.stats import rankdata
    import pandas as pd
    import numpy as np
    
    # making dataset that will store training data, PS: columns names must be inserted manually
    columns = ['dataset','n_clusters']+[f'MF{i}' for i in range(1,20)]+['Ward_Agglomerative_Clustering', 'Complete_Agglomerative_Clustering', 'Average_Agglomerative_Clustering', 'Single_Agglomerative_Clustering','K_Means','Mini_Batch_K_Means', 'Spectral_Clustering']
    metadata = pd.DataFrame(columns = columns)    

    # calculating and performace of each dataset
    i = 0
    for dataset in database:
        i+=1
        
        # Transforming pandas dataset to numpy array fo speed
        X = dataset.to_numpy()
        
        # extracting metafeatures:
        MF = list(Feature_Extractor(X, feature_type))
        
        # looping through diferent values of k 
        for k in k_range:
            
            # Getting algoritm performace
            CE_CH = Clustering_Evaluation(X, k, metric = calinski_harabasz_score, method = 'max', random_state = 0)
            CE_SS = Clustering_Evaluation(X, k, metric = silhouette_score,        method = 'max', random_state = 0)
            CE_DB = Clustering_Evaluation(X, k, metric = davies_bouldin_score,    method = 'min', random_state = 0)
            
            # Ranking algoritm performace
            rank_CH = np.array(list(CE_CH['Rank']))
            rank_SS = np.array(list(CE_SS['Rank']))
            rank_DB = np.array(list(CE_DB['Rank']))
            avg_rank = (rank_CH+rank_SS+rank_DB)/3
            rank = list(rankdata(avg_rank, method = 'ordinal'))
            
            # adding to dataframe
            metadata.loc[len(metadata)] = [i,k] + MF + rank
            
            print(f'i={i}, k={k}')
            
    metadata['n_clusters'] = metadata['n_clusters'].astype(int)
    metadata['dataset'] = metadata['dataset'].astype(int)
    return metadata


    


