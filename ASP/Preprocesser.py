def load_database_arff(path, save_in = None):
    from scipy.io import arff
    import pandas as pd
    import os

    # empty couter and list
    i=0
    database = []

    #find directories in givem path
    dir_list = os.listdir(path)
    
    # cicle through every file in directory
    for dataset in dir_list:
        
        # try to load any file in the path as an .arff dataset
        try:
            data = arff.loadarff(path+dataset)
        except:
            raise NameError(f"{path+dataset} could not be loaded.")
            continue
        
        # making dataframe from the data
        df = pd.DataFrame(data[0])
        [n, m] = df.shape
        
        # renaming columns
        df.columns = [f'col{i}' for i in range(m)]
        df.rename(columns={f'col{m-1}': "target"}, inplace=True)
        
        # replacing missing value b'?' with more identifiable nan
        df = df.replace(b'?', float('nan'))
        
        #if a save directory is provided, save the dataframe as csv file
        if save_in != None:
            i+=1
            df.to_csv(save_in + f'raw_dataset_{i:03d}.csv', index = False)
            
        print(f'raw_dataset_{i:03d}.csv <- {dataset}')

        # adding dataframe to database
        database.append(df)
    return database



def load_database(path):
    import pandas as pd
    import os

    #find directories in givem path
    dir_list = os.listdir(path)
    
    # cicle through every file in directory
    database = []
    for dataset in dir_list:
        
        # try to load any file in the path as an .csv dataframe
        try:
            df = pd.read_csv(path+dataset)
        except:
            raise NameError(f"{path+dataset} could not be loaded.")
            continue
        
        # adding dataframe to database
        database.append(df)
        
        print(dataset + ' loaded')
    
    return database



def preprocess_database(database, save_in = None, size_limit = None, nan_limit = 0):
    
    i=0
    new_database = []
    
    # looping through the datasets in the database
    for df in database:
        
        # droping the last column, as the target will be ignored
        df = df.drop(df.columns[-1],  axis=1)
        
        # separating data from the dataset
        [n, m] = df.shape

        # Truncating big datasets
        if size_limit!=None:
            if n*m > size_limit:
                cut = int(size_limit/m) - 1
                df = df.truncate(after=cut)
        
        # looping through the columns of the dataset
        for col in df.columns:
            
            # check if percentage of missing value in columns excedes the limit
            # if so drop the column and continue
            nan = df[col].isna().sum()/n
            if nan >= nan_limit:
                df = df.drop(col, axis=1)
                continue
            
            # check if all values in a column are the same
            # if so drop the column and continue
            if (df[col] == df[col][0]).all():
                df = df.drop(col, axis=1)
                continue
                
            # if a column can be converted to float, it will
            try:
                df[col] = df[col].astype(float)
                
                # replacing all nan values with mean value of the column
                df[col] = df[col].fillna(df[col].mean())
            
            # else a column is categorical (can't be converted to float), it will be encoded as integers
            except: 
                # enumarating every category and fliping the dictionary
                my_dict = dict(enumerate(set(df[col])))
                my_dict = {i:k for k,i in my_dict.items()}
                
                # for every index, the column value will be shifted to a integer acording to the dictionary
                df[col] = df[col].replace(my_dict)
                
                # replacing all nan values with the mode of the collumn
                df[col] = df[col].fillna(df[col].mode()[0])               
            
            # normalizing the columns
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
            
        # Deleting rows that are constant
        to_delete = []
        for row in range(len(df)):
            the_row_is_constant = (df.iloc[row] == df.iloc[row,0]).min()
            if the_row_is_constant == True:
                to_delete.append(row)
        df = df.drop(to_delete, axis=0)
            
        #if a save directory is provided, save the dataframe as csv file
        if save_in != None:
            i+=1
            df.to_csv(save_in + f'dataset_{i:03d}.csv', index = False)
        
        print(f'dataset_{i:03d}.csv created')

        #making and returning a database
        new_database.append(df)
    return new_database



def preprocess_data(df, size_limit = None, nan_limit = 0):
        
    # droping the last column, as the target will be ignored
    df = df.drop(df.columns[-1],  axis=1)
        
    # separating data from the dataset
    [n, m] = df.shape
    # Truncating big datasets
    if size_limit!=None:
        if n*m > size_limit:
            cut = int(size_limit/m) - 1
            df = df.truncate(after=cut)
        
    # looping through the columns of the dataset
    for col in df.columns:

    # check if percentage of missing value in columns excedes the limit
    # if so drop the column and continue
        nan = df[col].isna().sum()/n
        if nan >= nan_limit:
            df = df.drop(col, axis=1)
            continue
            
        # check if all values in a column are the same
        # if so drop the column and continue
        if (df[col] == df[col][0]).all():
            df = df.drop(col, axis=1)
            continue
                
        # if a column can be converted to float, it will
        try:
            df[col] = df[col].astype(float)
                
            # replacing all nan values with mean value of the column
            df[col] = df[col].fillna(df[col].mean())
            
        # else a column is categorical (can't be converted to float), it will be encoded as integers
        except: 
            # enumarating every category and fliping the dictionary
            my_dict = dict(enumerate(set(df[col])))
            my_dict = {i:k for k,i in my_dict.items()}
                
            # for every index, the column value will be shifted to a integer acording to the dictionary
            df[col] = df[col].replace(my_dict)
                
            # replacing all nan values with the mode of the collumn
            df[col] = df[col].fillna(df[col].mode()[0])               
            
        # normalizing the columns
        df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

    return df