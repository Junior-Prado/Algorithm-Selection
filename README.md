# ASP
Functions dedicated to The Algoritm Selection problem with a focus on a metalearning approach


On the preprocessing file there are three functions dedicated to prepare datasets to be used in later functions. Being then:

load_database_aff: that as made to load OpenML datasets.
load_database: that loads a batch of csv files into a list of pandas dataframes.
preprocess_database: that modify a list of datasets into a format susceptible to extract metafeatures

On the trainingdata file there is a class and two function, dedicated to extracting metafeatures and rankings of datasets. Being then:

Feature_Extractor: function that takes metafeatures based on distance and correlation described in [inset reference]
Clustering Evaluation: class that compares and rank the results of clustering performed in a dataset according to seven clustering algorithms of sklearn. It's useful on its own independent of other function on this repository
rank_database: apply the other functions on this file to all datasets in a database and return prediction data that can be used to predict the ranking of unknown datasets. It is very slow but needsto be only used once

Note1: rank database is useful if you have your own database and plans to expand it in the futures. The more coherent a database is the better the results of the predictions will be. For example: if your database is made of health data to predict if a person is susceptible to a hearth attack, the resulting data will be good at prediction exactly that for new datasets. but if you try to use that data to predict the ranking of a car crash dataset, your predictions will be of mark.

Note2: test_data.csv is a precomputed prediction data made with a mix of real datasets that can be used to make your predictions and skip this whole process.

On the metalearnig file there are four function, dedicated to making prediction based on metadata. Being then:

split_test_data: auxiliary function that can split a prediction data into training and testing data.
NN_weigth: auxiliary function that will use nearest neighbours to compare and metafeatures data of known data sets with a unkown one and make a weight vector based on it
predicting_rank: will use NN_weigth and prediction data 'X' to predict the ranking of an dataset with metafeatures 'y'
get_all_predictions: is a function that will split the prediction data n times and make predictions on the ranking of each instance and calculate correlation of each one with its true ranking

Note3: get_all_predictions is useful to test how well your own prediction database is performing and find a good value of the number of neighbours that works for many datasets.
Note4: all_scores.csv is the result of applying get_all_predictions to test_data.csv

