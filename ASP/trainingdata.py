def Feature_Extractor(X):
    
    import numpy as np
    from scipy.stats import spearmanr, skew, kurtosis, zscore, rankdata
    
    # dataset format
    [n, p] = np.shape(X)

    # Ranking of instances
    R = rankdata(X, method = 'dense', axis=0)
    
    # Declaring empty vectors
    size = n*(n-1)-1
    correlation =  np.zeros(int(size/2+1))
    dissimilarity =  np.zeros(int(size/2+1))
    Meta = np.zeros(size)
    
    # calculating correlation and dissimilarity
    i=0
    for k in range(n):
        for l in range(k+1,n):
            
            # checking if an array is constant, if so modify it
            if np.all(R[k] == R[k][0]):
                R[k][0] +=1
            if np.all(R[l] == R[l][0]):
                R[l][0] +=1
                
            correlation[i] = spearmanr(R[k],R[l]).correlation
            dissimilarity[i] = np.linalg.norm(X[k] - X[l])
            i+=1
    
    # concatenating the correlation and dissimilarity into a metadata vector
    meta = np.concatenate([correlation,dissimilarity])
    
    # normalizing the metadata vector
    for i in range(n*(n-1)-1):
        Meta[i] = (meta[i] - np.min(meta)) / (np.max(meta) - np.min(meta))
        
    # calculating zscore from metadata array
    zs_meta = zscore(Meta)
    size_z = np.count_nonzero((zs_meta >= 0.0))
    
    # calculating metafeatures
    MF1 = np.mean(Meta)
    MF2 = np.var(Meta)
    MF3 = np.std(Meta)
    MF4 = skew(Meta)
    MF5 = kurtosis(Meta)
    
    MF6  = np.count_nonzero((Meta >=  0) & (Meta <= 0.1))/size
    MF7  = np.count_nonzero((Meta > 0.1) & (Meta <= 0.2))/size
    MF8  = np.count_nonzero((Meta > 0.2) & (Meta <= 0.3))/size
    MF9  = np.count_nonzero((Meta > 0.3) & (Meta <= 0.4))/size
    MF10 = np.count_nonzero((Meta > 0.4) & (Meta <= 0.5))/size
    MF11 = np.count_nonzero((Meta > 0.5) & (Meta <= 0.6))/size
    MF12 = np.count_nonzero((Meta > 0.6) & (Meta <= 0.7))/size
    MF13 = np.count_nonzero((Meta > 0.7) & (Meta <= 0.8))/size
    MF14 = np.count_nonzero((Meta > 0.8) & (Meta <= 0.9))/size
    MF15 = np.count_nonzero((Meta > 0.9) & (Meta <= 1.0))/size
    
    MF16 = np.count_nonzero((zs_meta >= 0.0) & (zs_meta < 1.0))/size_z
    MF17 = np.count_nonzero((zs_meta >= 1.0) & (zs_meta < 2.0))/size_z
    MF18 = np.count_nonzero((zs_meta >= 2.0) & (zs_meta < 3.0))/size_z
    MF19 = np.count_nonzero((zs_meta >= 3.0))/size_z

    MF = np.array([MF1, MF2, MF3, MF4, MF5, MF6, MF7, MF8, MF9, MF10, MF11, MF12, MF13, MF14, MF15, MF16, MF17, MF18, MF19])
    
    return MF

class Clustering_Evaluation:
    
    # Class that takes a parameters of varios clustering algoritms and returns the performace ranked
    
    # 'k' is an integer describing the number of clusters
    # 'metric' is an function describing the scoring method, the metric must take a dataset and a labels only 
    # 'method' is a stirng that decides the ranking method, where:
    #     'max' means higher score
    #     'min' means minimun score
    #     'absolute' means score closer to zero
    # 'random_state' is the seed for algoritms that need it
    
    # the Methods on the class are:
    # as_frame, takes X and return a dataframe with the clustering and ranking described in the class
    # as_print, takes X and return text with the clustering and ranking described in the class
    # as_list, takes X and return a list with the clustering and ranking described in the class

    
    def __init__ (self, k=2, metric=None, method='max', random_state = None):
        self.k = k
        self.metric = metric
        self.method = method
        self.random_state = random_state
    pass
    
    def rank(self,X):
        
        # Importing functions
        import numpy as np
        from scipy.stats import rankdata
        from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, SpectralClustering, Birch
        
        if self.random_state == None:
            from random import randint
            self.random_state == randint(0,1000)
    
        # Running clustering algoritms and saving objects on a dictionary
        clusters_dict = {
            'Ward_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = self.k, linkage = 'ward').fit(X).labels_,
            'Complete_Agglomerative_Clustering':AgglomerativeClustering(n_clusters = self.k, linkage = 'complete').fit(X).labels_,
            'Average_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = self.k, linkage = 'average').fit(X).labels_,
            'Single_Agglomerative_Clustering': AgglomerativeClustering(n_clusters = self.k, linkage = 'single').fit(X).labels_,
            'K_Means':KMeans(n_clusters = self.k, random_state = self.random_state).fit(X).labels_,
            'Mini_Batch_K_Means': MiniBatchKMeans(n_clusters = self.k, random_state = self.random_state).fit(X).labels_,
            'Spectral_Clustering': SpectralClustering(n_clusters = self.k).fit(X).labels_
        }
    
        # Extracting algorithms_names from dictionary and making empty score list
        algorithms_names = list(clusters_dict.keys())
        scores = []

        # scoring each clustering method acording to chosen metric function
        # if none default to silhouette score   
        
        elif self.metric == None:
            from sklearn.metrics import silhouette_score
            for key, labels in clusters_dict.items():
                scores.append(silhouette_score(X, labels))
        else:
            for key, labels in clusters_dict.items():
                scores.append(self.metric(X, labels))
    
        # deciding the ranking method
        if (self.method == 'min') or (self.method == 'max'):
            ranked_scores = rankdata(scores, method = 'ordinal')
        
            if self.method == 'max':
                ranked_scores = len(clusters_dict)+1-ranked_scores  
            
        elif (self.method == 'absolute'):
            abs_scores = np.absolute(scores)
            ranked_scores = rankdata(abs_scores, method = 'ordinal')
        
        else:
            raise ValueError("Error: 'method' must be 'min', 'max' or 'absolute") 
        
        return ranked_scores, algorithms_names, scores
    
    # method that return a dataframe with the results
    def as_frame(self, X):
        import pandas as pd

        # ranking algoritms
        ranked_scores, algorithms_names, scores = self.rank(X)
        
        # making and sorting dataframe
        frame = pd.DataFrame({
            'Rank':ranked_scores,
            'Name':algorithms_names,
            'Score':scores
            })
        frame = frame.sort_values(by = 'Rank')
        frame.set_index('Rank', inplace=True)
        return frame
    
    # method that prints the results
    def as_print(self, X):
            import numpy as np
            
            # ranking algoritms
            ranked_scores, algorithms_names, scores = self.rank(X)
            
            # printing rankings
            print("\033[1m" + "Best Algorithms Ranked:" + "\033[0m")
            for i in range(8):
                index = np.where(ranked_scores==i+1)[0][0]
                print(f"{i+1}° {algorithms_names[index]}: {scores[index]}")
            
            return None
    
    # method that return a list with the results
    def as_list(self, X):
            # ranking algoritms
            ranked_scores, algorithms_names, scores = self.rank(X)
        
            # making, transposing and returning performace list
            performace = [ranked_scores.tolist(),algorithms_names,scores]
            #performace = list(map(list, zip(*performace)))
            return performace

def rank_database(database, min_k = 2, max_k = 11): 
    from scipy.stats import rankdata
    import pandas as pd
    import numpy as np
    
    # making dataset that will store training data, PS: columns names must be inserted manually
    MF_names = ['MF1', 'MF2', 'MF3', 'MF4', 'MF5', 'MF6', 'MF7', 'MF8', 'MF9', 'MF10', 'MF11', 'MF12', 'MF13', 'MF14', 'MF15', 'MF16', 'MF17', 'MF18', 'MF19']
    AL_names = ['Ward_Agglomerative_Clustering', 'Complete_Agglomerative_Clustering', 'Average_Agglomerative_Clustering', 'Single_Agglomerative_Clustering','K_Means','Mini_Batch_K_Means', 'Spectral_Clustering']
    training_data = pd.DataFrame(columns = ['k'] + MF_names + AL_names)
    
    # calculating metafeatures and performace of each dataset
    i = 0
    for dataset in database:
        i+=1
        
        # Separating data from target
        X = dataset.iloc[:,0:-1].to_numpy()
        
        # extracting metafeatures:
        MF = list(Feature_Extractor(X))
        
        # looping through diferent values of k 
        for k in range(min_k,max_k):
            print(f'i={i}, k={k}')
            
            # Ranking algoritm performace
            from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
            CE_CH = Clustering_Evaluation(k=k, metric = calinski_harabasz_score, method = 'max', random_state = 0)
            CE_SS = Clustering_Evaluation(k=k, metric = silhouette_score,        method = 'max', random_state = 0)
            CE_DB = Clustering_Evaluation(k=k, metric = davies_bouldin_score,    method = 'min', random_state = 0)
            rank_CH = np.array(CE_CH.as_list(X)[0])
            rank_SS = np.array(CE_CH.as_list(X)[0])
            rank_DB = np.array(CE_CH.as_list(X)[0])
            avg_rank = (rank_CH+rank_SS+rank_DB)/3
            rank = rankdata(avg_rank, method = 'ordinal')
            
            # adding to dataframe
            data = [k]+MF+list(rank)
            training_data.loc[len(training_data)] = data

    return training_data