# standard libraries
import pandas as pd
import numpy as np

# iterate over dataframes
import itertools

# visualization library
import matplotlib.pyplot as plt

# distance calculations
from sklearn.neighbors import BallTree

# Random Forest model
from sklearn.ensemble import RandomForestClassifier


# dummy yes/no columns
def yes_to_one(df, cols):
    
    '''
    Turn columns with 'Yes' and 'No' values into 1s and 0s.
    
    Input
    -----
    df : Pandas DataFrame
    cols : list, column names
    
    Output
    ------
    NONE : Overwrites the input columns!
    '''
    
    # replace values with 0 or 1 for each column in list
    for col in cols:
        df[col] = np.where(df[col] == 'Yes', 1, 0)
        

# find outliers
def find_extremes(df, num_std):
    
    '''
    Function to find columns that contain outlier values.
    
    Input
    -----
    df : Pandas DataFrame
    num_std : int, number of standard deviations from the mean
    
    Output
    ------
    extreme_list : list, columns that contain outlier values
    '''
    
    # instantiate empty columns list
    extreme_list = []
    
    # loop over all columns in dataframe
    for column in list(df.columns):
        
        # add columns greater than std_val standard deviations from the mean
        if df[column].max() > (df[column].mean() + std_val*df[column].std()):
            extreme_list.append(column)
            
        # add columns less than num_std standard deviations from the mean
        if df[column].min() < (df[column].mean() - std_val*df[column].std()):
            extreme_list.append(column)
    
    # output columns list
    return extreme_list


# control value of outliers
def rein_extremes(df, columns, num_std):
    
    '''
    Function to replace outlier values using a multiple of the data's
    standard deviation.
    
    Input
    -----
    df : Pandas DataFrame
    columns : list, columns to transform
    num_std : int, number of standard deviations from the mean
    
    Output
    ------
    NONE : Overwrites the input columns!
    '''
    
    # loop over columns list
    for column in columns:
        
        # find mean and standard deviation of column
        mean = df[column].mean()
        std = df[column].std()
        
        # find values less than or greater than num_std standard deviations from the mean
        conditions = [df[column] > mean + num_std*std,
                      df[column] < mean - num_std*std]
        
        # replace with num_std standard deviations from the mean
        choices = [mean + num_std*std,
                   mean - num_std*std]
        
        df[column] = np.select(conditions, choices, df[column])
        

# confusion matrix plotter
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """
    This function prints and plots a model's confusion matrix.
    
    Input
    -----
    cm : sklearn confusion matrix, `sklearn.metrics.confusion_matrix(y_true, y_pred)`
    classes : list, names of target classes 
    
    Optional input
    --------------
    normalize : bool, whether to apply normalization (default=False)
            Normalization can be applied by setting `normalize=True`.
    title : str, title of the returned plot
    cmap : matplotlib color map
    
    Output
    ------
    Prints a stylized confusion matrix.
    
    
    [Code modified from work by Sean Abu Wilson.]
    """
    
    # convert to percentage, if normalize set to True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # format true positives and others
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # add axes labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    

# random forest feature importances plotter
def plot_forest_features(model, X, num_features=15, to_print=True):
    
    """
    This function plots feature importances for Random Forest models and
    optionally prints  a dictionary with features and their measure of
    importance.
    
    Input
    -----
    model : Random Forest model (`sklearn.ensemble.RandomForestClassifier()`)
    X : Pandas DataFrame, features used in model
    
    Optional input
    --------------
    num_features : int, the number of features to plot/print (default=15)
            All feature importances can be shown by setting
            `num_features=X.shape[1]`.
    to_print : bool, whether to print list of feature names and their
               impurity decrease values
            Printing can be turned off by setting `to_print=False`.
    
    Output
    ------
    Prints a bar graph and optional list of tuples.
    
    
    [Code modified from work by Sean Abu Wilson.]
    """
    
    # list of tuples (column index, measure of feature importance)
    imp_forest = model.feature_importances_
    
    # sort feature importances in descending order, slicing top number of features
    indices_forest = np.argsort(imp_forest)[::-1][:num_features]

    # rearrange feature names so they match the sorted feature importances
    names_forest = [X.columns[i] for i in indices_forest]

    # create plot, using num_features as a dimensional proxy 
    plt.figure(figsize=(num_features, num_features/1.5))
    plt.bar(range(num_features), imp_forest[indices_forest])

    # prettify plot
    plt.title('Random Forest Feature Importances', fontsize=30, pad=15)
    plt.ylabel('Average Decrease in Impurity', fontsize=22, labelpad=20)
    # add feature names as x-axis labels
    plt.xticks(range(num_features), names_forest, fontsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)

    # Show plot
    plt.tight_layout()
    plt.show()
    
    if to_print:
        # print a list of feature names and their impurity decrease value in the forest
        print([(i,j) for i,j in zip(names_forest, imp_forest[indices_forest])])
    
        
## The following code was copied from 
## It has been edited slightly to suit my purposes.
# calculate distance between two points
def get_nearest(src_points, candidates, k_neighbors=3):
    
    '''
    Function to calculate distances between points, comparing two lists. 
    
    Input
    -----
    src_points : list, location data (e.g., latitude and longitude)
    candidates : list, location data to compare with src_points
            (e.g., latitude and longitude)
    
    Optional input
    --------------
    k_neighbors : int, number of closest points to find (default=3)
    
    Output
    ------
    tuple :
        target_tree : indices of nearest points
        closest_dist : distance to nearest neighbor (in meters)
        
        
    [Code modified from:
     https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html]
    '''
    
    # NOTE: haversine distance which is implemented here is a bit slower than using e.g. 'euclidean' metric
    # but useful as we get the distance between points in meters.

    # create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # keep index of tree whose neighbor is being found and distance to the neighbor
        # NOTE: since I'm running this within the same dataframe, take distances 1,
        # since distances[0] will have a distance of 0, it being the same point.
    target_tree = indices[0]
    closest_dist = distances[1]

    # output indices and distances
    return (target_tree, closest_dist)

def nearest_neighbor(gdf, name=None):
    
    """
    Function to find the nearest point within same GeoDataFrame.
    
    Input
    -----
    gdf : GeoPandas GeoDataFrame
        Assumes that the input Points are in WGS84 projection 
        (latitude and longitude).
    
    Optional input
    --------------
    name : str, name for the returned Pandas Series (default=None)
    
    Output
    ------
    distances : Pandas Series
    
    
    [Code modified from:
     https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html]
    """
    
    # parse coordinates from points and insert them into a numpy array as RADIANS
    radians = np.array(gdf['geometry'].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # find the nearest points
    # -----------------------
    # tree ==> index that corresponds to the tree whose closest neighbor is being measured
    # neighbor_dist ==> distance to tree's closest neighbor (in meters)

    tree, neighbor_dist = get_nearest(src_points=radians, candidates=radians)

    # convert to meters from radians into a pandas series
    earth_radius = 6371000
    distances = pd.Series(neighbor_dist * earth_radius, name=name)

    # output geodataframe with new column
    return distances


# custom scoring function
def good_precision(y_true, y_pred, **kwargs):
    
    '''
    Custom scoring function calculating precision of 'Good' predictions.
    
    Input
    -----
    y_true : Pandas Series or NumPy array, actual labels
    y_pred : Pandas Series or NumPy array, predicted labels
    
    Output
    ------
    float : precision score for 'Good' predictions
    '''
    
    # true positives
    t_p = 0
    # false positives
    f_p = 0
    
    # loop over indices
    for i in range(len(y_true)):
        # only if prediction value is 'Good'
        if y_pred[i] == 'Good':
            # add to true positives count
            if y_true.iloc[i] == 'Good':
                t_p += 1
                
            # add to false positives count
            else:
                f_p += 1
                
    # output precision calculation
    return t_p / (t_p + f_p)