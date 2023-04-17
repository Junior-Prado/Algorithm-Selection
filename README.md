# ASP
Functions dedicated to The Algoritm Selection problem with a focus on a metalearning approach

On the Preprocesser file there are three functions dedicated to prepare datasets to be used in later functions. Being then:

load_database_arff: that as made to load OpenML datasets.

load_database: that loads a batch of csv files into a list of pandas dataframes.

preprocess_database: that modify a list of datasets into a format susceptible to extract metafeatures.

preprocess_database: that modify a simgle datasets into a format susceptible to extract metafeatures.


On the MakeTrainingData file there is a class and two function, dedicated to extracting metafeatures and rankings of datasets. Being then:

Feature_Extractor: function that takes metafeatures based on distance and correlation described in [inset reference]

Clustering Evaluation: class that compares and rank the results of clustering performed in a dataset according to seven clustering algorithms of sklearn. It's useful on its own independent of other function on this repository.

metadata_extractor: apply the other functions on this file to all datasets in a database and return metadata that can be used to predict the ranking of unknown datasets. It's slow but needs to be only used once per database.


Note1: metadata_extractor is useful if you have your own database and plans to expand it in the futures. The more coherent a database is the better the results of the predictions will be. For example: if your database is made of health data to predict if a person is susceptible to a hearth attack, the resulting data will be good at prediction exactly that for new datasets. but if you try to use that data to predict the ranking of a car crash dataset, your predictions will be of mark.

Note2: CaD.csv and distance.csv are precomputed prediction data made with a mix of real datasets that can be used to make your predictions and skip the process of making on yourself.


On the MetaLearner file there are five function, dedicated to making prediction based on metadata. Being then:

NN_weigth: auxiliary function that will use nearest neighbours to compare and metafeatures data of known data sets with a unkown one and make a weight vector based on it.

predicting_rank: will use NN_weigth and metadata 'X' to predict the ranking of an dataset with metafeatures 'y'.

get_all_predictions: is a function that will compare each dataset of a database with itself and predictic the ranking of each instance, then and calculate correlation of each prediction with it's true ranking.

load_CaD: equivalent of loading the CaD.csv to a pandas dataframe.

load_distance: equivalent of loading the load_distance.csv to a pandas dataframe.


Note3: if predicting_rank does not receive a metadata file it will use load_CaD as the metadata.

Note4: get_all_predictions is useful to test how well your own prediction database is performing and find a good value of the number of neighbours that works for many datasets.
