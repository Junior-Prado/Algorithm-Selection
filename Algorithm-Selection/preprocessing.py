from scipy.io import arff
import pandas as pd
import os

def load_database_arff(path, save_in = None, there_is_target_data = True):
    # Function for batch loading .ARFF files, like the datasets from OpenML
    # it takes the following parameters:
    # path (string) -> parameter with the path to a folder full of .ARFF files
    # save_in (string) -> optional parameter with the path to a folder to save datasets as .csv files
    # there_is_target_data (bool) -> optional parameter with that ask if the files comes with target data
    # and returns a list of dataframes

    file_index=0
    database = []
    list_of_directories = os.listdir(path)
    
    for file in list_of_directories:
        
        #if a file is in the . arff format, load its data, else ignore it      
        if os.path.splitext(file)[1] == ".arff":                 
            data = arff.loadarff(path+file)
        else:
            raise NameError(f"{path+file} could not be loaded, as it is not a .arff file.")
            continue
    
        dataset = pd.DataFrame(data[0])                                   # making a dataframe from the data in the file
        [n, m] = dataset.shape                                            # storing shape of dataset
        dataset.columns = [f'col{i}' for i in range(m)]                   # renaming columns of the dataset to generic columns names
        if there_is_target_data:
            dataset.rename(columns={f'col{m-1}': "target"}, inplace=True) # renaming the last column to 'target'
        dataset = dataset.replace(b'?', float('nan'))                     # replacing missing value b'?' for a more identifiable nan
        
        #if a save directory is provided, save the dataframe as .csv file               
        if save_in != None:                                
            file_index+=1
            dataset.to_csv(save_in + f'raw_dataset_{file_index:03d}.csv', index = False)
            print(f'raw_dataset_{file_index:03d}.csv <- {file}')
        
        database.append(dataset)
    return database



def load_database(path):
    # Function for batch loading .csv files, it takes the parameter:
    # path (string) -> parameter with path to a folder full of .csv files
    # and returns a list of dataframes

    list_of_directories = os.listdir(path)
    database = []
    
    for file in list_of_directories:
        
        if os.path.splitext(file)[1] == ".csv":
            dataset = pd.read_csv(path+file)
        else:
            raise NameError(f"{path+file} could not be loaded, as it is not a .csv file.")
            continue
        
        print(file + ' loaded')
        database.append(dataset)
    return database



def preprocess_database(database, save_in = None, size_limit = None, nan_limit = 0, there_is_target_data = True):
    # Function for batch preprocessing a list of datasets (in dataframe format) to get then ready for
    # training a metaleaning system, each dataset will then pass through the following process:
    #
    #    1. Removal of all labels.
    #    2. Datasets with more than "size_limit" cells will be truncated.
    #    3. columns with more than "nan_limit" missing data will be removed.
    #    4. Removal of contant columns.
    #    5. All categorical data will be converted to integers.
    #    6. Missing numerical values will be replaced by the average of the columns.
    #    7. Missing categorical values will be replaced by the column's mode.
    #    8. All attributes were normalized in the interval [0,1].
    #
    # The function takes the following parameter:
    # database (list) -> parameter with the list of datasets.
    # save_in (string) -> optional parameter with path to a folder to save datasets as .csv files.
    # size_limit (int) -> optional parameter with the maximum number of cells.
    # size_limit (float) -> optional parameter with the tolerable percentage of missing values per parameter.
    # there_is_target_data (bool) -> optional parameter with that ask if the database comes with target data.
    # 
    # the function returns a list of processed dataframes
    
    dataset_index=0
    new_database = []
    
    for dataset in database:
        
        #dropping the target column, if it is provided
        if there_is_target_data:
            dataset = dataset.drop(dataset.columns[-1],  axis=1) 
        
        [n, m] = dataset.shape  # storing shape of the dataset

        # Truncating big datasets
        if size_limit!=None: 
            if n*m > size_limit:
                cut = int(size_limit/m) - 1
                dataset = dataset.truncate(after=cut)

        for col in dataset.columns:
            
            # check if percentage of missing value in column excedes the given limit, if so drop the column and continue
            if (dataset[col].isna().sum()/n) >= nan_limit:
                dataset = dataset.drop(col, axis=1)
                continue
            
            # check if all values in a column are the same, if so drop the column and continue
            if (dataset[col] == dataset[col][0]).all():
                dataset = dataset.drop(col, axis=1)
                continue
                
            # try to convert a column to float, if it fails, the a column is categorical and will be converted to integers
            try:
                dataset[col] = dataset[col].astype(float)                 # converting column to float
                dataset[col] = dataset[col].fillna(dataset[col].mean())   # replacing all nan values with mean value of the column
            
            except: 
                my_dict = dict(enumerate(set(dataset[col])))               # enumarating every category and saving it in a dictionary
                my_dict = {i:k for k,i in my_dict.items()}                 # fliping the dictionary
                dataset[col] = dataset[col].replace(my_dict)               # replacing every category with an integer acording to the dictionary
                dataset[col] = dataset[col].fillna(dataset[col].mode()[0]) # replacing all nan values with the mode of the column       
            
            dataset[col] = (dataset[col]-dataset[col].min())/(dataset[col].max()-dataset[col].min()) # normalizing the columns
            
        #if a save directory is provided, save the dataframe as .csv file
        if save_in != None:
            dataset_index+=1
            dataset.to_csv(save_in + f'dataset_{dataset_index:03d}.csv', index = False)
            print(f'dataset_{dataset_index:03d}.csv created')
            
        new_database.append(dataset)
    return new_database



def preprocess_data(dataset, size_limit = None, nan_limit = 0, there_is_target_data = True):
    # Function for preprocessing a single of datasets (in dataframe format) to get then ready for
    # a metaleaning system, the dataset will then pass through the following process:
    #
    #    1. Removal of all labels.
    #    2. Datasets with more than "size_limit" cells will be truncated.
    #    3. columns with more than "nan_limit" missing data will be removed.
    #    4. Removal of contant columns.
    #    5. All categorical data will be converted to integers.
    #    6. Missing numerical values will be replaced by the average of the columns.
    #    7. Missing categorical values will be replaced by the column's mode.
    #    8. All attributes were normalized in the interval [0,1].
    #
    # The function takes the following parameter:
    # dataset (dataframe) -> parameter with the list of datasets.
    # size_limit (int) -> optional parameter with the maximum number of cells.
    # size_limit (float) -> optional parameter with the tolerable percentage of missing values per parameter.
    # there_is_target_data (bool) -> optional parameter with that ask if the database comes with target data.
    # 
    # the function return the processed dataframe
        
    #dropping the target column, if it is provided
    if there_is_target_data:
        dataset = dataset.drop(dataset.columns[-1],  axis=1) 
        
    [n, m] = dataset.shape  # storing shape of the dataset

    # Truncating big datasets
    if size_limit!=None: 
        if n*m > size_limit:
            cut = int(size_limit/m) - 1
            dataset = dataset.truncate(after=cut)
        
    # looping through the columns of the dataset
    for col in dataset.columns:
            
        # check if percentage of missing value in column excedes the given limit, if so drop the column and continue
        if (dataset[col].isna().sum()/n) >= nan_limit:
            dataset = dataset.drop(col, axis=1)
            continue
            
        # check if all values in a column are the same, if so drop the column and continue
        if (dataset[col] == dataset[col][0]).all():
            dataset = dataset.drop(col, axis=1)
        continue
                
        # try to convert a column to float, if it fails, the a column is categorical and will be converted to integers
        try:
            dataset[col] = dataset[col].astype(float)                 # converting column to float
            dataset[col] = dataset[col].fillna(dataset[col].mean())   # replacing all nan values with mean value of the column
            
        except: 
            my_dict = dict(enumerate(set(dataset[col])))               # enumarating every category and saving it in a dictionary
            my_dict = {i:k for k,i in my_dict.items()}                 # fliping the dictionary
            dataset[col] = dataset[col].replace(my_dict)               # replacing every category with an integer acording to the dictionary
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0]) # replacing all nan values with the mode of the column       

        dataset[col] = (dataset[col]-dataset[col].min())/(dataset[col].max()-dataset[col].min()) # normalizing the columns

    return dataset
