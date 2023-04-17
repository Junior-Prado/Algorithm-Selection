def split_train_test(data, k = 2, index = 0):

    # isolando dataset para dado valor de k
    data = data[data['n_clusters'] == k]
    data = data.reset_index()
    data = data.iloc[:,2:]
    
    # isolando dados de teste
    X = data.drop(index, axis = 0)
    X_MF = X.iloc[:,0:19]
    X_RK = X.iloc[:,19:26]
    
    y = data.iloc[index]
    y_MF = y.iloc[0:19]
    y_RK = y.iloc[19:26]
    
    return X_MF, X_RK, y_MF, y_RK

def NN_weights(X,y,n_neighbors = 6):
    from sklearn.neighbors import KDTree
    import numpy as np
    
    # initiation
    [n, m] = X.shape
    weights = np.zeros(n)
    
    #creating tree
    tree = KDTree(X)  
    distance, index = tree.query([y], k=n_neighbors)

    j = 0
    for i in index:
        weights[i] = distance[j]
        j+=1
    return weights/weights.sum()

def get_ranking(X,y_MF,n_neighbors = 6):
    from scipy.stats import rankdata
    import pandas as pd
            
    # separating prediction data in  meta data and rank data
    X_MF = X.iloc[:,0:19]
    X_RK = X.iloc[:,19:26]

    # predicting weight by comparing the test data with the training data
    weights = NN_weights(X_MF,y_MF,n_neighbors)
    rank_score = weights.dot(X_RK)
    rank = rankdata(rank_score, method = 'ordinal')

    # adding result to a dataframe and returning it
    df = pd.DataFrame(columns = ['Ward_Agglomerative_Clustering', 'Complete_Agglomerative_Clustering', 'Average_Agglomerative_Clustering', 'Single_Agglomerative_Clustering','K_Means','Mini_Batch_K_Means', 'Spectral_Clustering'])
    df.loc[len(df)] = rank
    
    return df

def get_all_scores(data):
    from scipy.stats import spearmanr, rankdata
    import pandas as pd
    
    
    # geting the min and max values of k and the number of datasets
    min_k = int(data['n_clusters'].min())
    max_k = int(data['n_clusters'].max()+1)  
    N = (data['n_clusters'] == min_k).to_numpy().sum()
    

    # making columns of the new dataframe and declaring it
    columns = ['Test_Dataset','N_Neighbors']
    for k in range(min_k,max_k):
        columns.append(f'k = {k}')
    all_scores = pd.DataFrame(columns = columns)
    
    for index in range(N):
        for n in range(2,N):
            
            # making new line and adding the Number of Neighbors value
            new_row = [int(index),int(n)]
            
            #looping throught k values
            for k in range(min_k,max_k):
                print(f'index = {index} - n = {n} - k = {k}')
                
                # separating only the data with same k value
                df = data[data['n_clusters'] == k]
                df = df.reset_index()
                df = df.iloc[:,2:]
                
                # separating train and test data, and metadata from rank data
                X = df.drop(index, axis = 0)
                X_MF = X.iloc[:,0:19]
                X_RK = X.iloc[:,19:26]
                y = df.iloc[index]
                y_MF = y.iloc[0:19]
                y_RK = y.iloc[19:26]
                
                # predicting weight by comparing the test data with the training data
                weights = NN_weights(X_MF,y_MF,n_neighbors = n)
                rank_score = weights.dot(X_RK)
                rank = rankdata(rank_score, method = 'ordinal')
                
                # calculating the corelation of actual test data and adding it to new row
                correlation = spearmanr(rank, y_RK).correlation
                new_row.append(correlation)
                
            # adding new row to dataframe
            all_scores.loc[len(all_scores)] = new_row
    
    all_scores['N_Neighbors'] = all_scores['N_Neighbors'].astype(int)
    all_scores['Test_Dataset'] = all_scores['Test_Dataset'].astype(int)
    return all_scores