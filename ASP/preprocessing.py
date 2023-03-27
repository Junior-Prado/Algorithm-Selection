def load_database_aff(path):
    from scipy.io import arff
    import pandas as pd
    import os

    dir_list = os.listdir(path)
    
    database = []
    for dataset in dir_list:
        new_path = path+dataset
        data = arff.loadarff(new_path)
        database.append(pd.DataFrame(data[0]))
    
    return database

def load_database(path):
    import pandas as pd
    import os
    
    dir_list = os.listdir(path)
    
    database = []
    for dataset in dir_list:
        database.append(pd.read_csv(f'database/{dataset}'))
    
    return database

def preprocess_database(database, limit = 5000):
    import pandas as pd

    i = 0
    j = 0
    new_database = []
    
    # looping through the datasets in the database
    for df in database:
        
        # separating data from the dataset
        [n, m] = df.shape
        
        # skiping datasets too big
        if n*m > limit:
            continue
            
        # renaming columns
        df.columns = [f'col{i}' for i in range(m)]
        df.rename(columns={f'col{m-1}': "target"}, inplace=True)
        
        # looping through the columns of the dataset
        for col in df.columns:      
            # if a column can be converted to float, it will
            try:
                df[col] = df[col].astype(float)
            
            # if a column is categorical (can't be converted to float), it will be encoded as integers
            except: 
                # enumarating every category and fliping the dictionary
                my_dict = dict(enumerate(set(df[col])))
                my_dict = {i:k for k,i in my_dict.items()}
                
                # for every index, the column value will be shifted to a integer acording to the dictionary
                df[col] = df[col].replace(my_dict)
                    
            # normalizing the column, except the last
            if col != df.columns[-1]:
                df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
            else:
                df[col] = df[col].astype(int)

        #making and returning a database
        new_database.append(df)
    return new_database