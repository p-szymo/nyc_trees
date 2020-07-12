import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

def yes_to_one(df, cols):
    '''Turn columns with 'Yes' and 'No' values into 1s and 0s.
    Overwrites the input columns!
    '''
    for col in cols:
        df[col] = np.where(df[col] == 'Yes', 1, 0)
        
def find_extremes(df):
    '''Takes in a dataframe and returns a list of columns with values farther than 4 standard deviations from the mean.'''
    extreme_list = []
    for column in list(df.columns):
        if df[column].max() > (df[column].mean() + 4*df[column].std()):
            extreme_list.append(column)
        if df[column].min() < (df[column].mean() - 4*df[column].std()):
            extreme_list.append(column)
    return extreme_list

def rein_extremes(df, columns):
    '''Takes in a dataframe and a list of columns and changes any values farther than 4 standard deviations from the mean
    to 4 standard deviations from the mean.
    Overwrites the input column!'''
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        conditions = [df[column] > mean + 4*std,
                      df[column] < mean - 4*std]
        choices = [mean + 4*std,
                   mean - 4*std]
        df[column] = np.select(conditions, choices, df[column])
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_forest_features(model, X, num_features=15, to_print=True):
    
    imp_forest = model.feature_importances_
    
    # sort feature importances in descending order
    indices_forest = np.argsort(imp_forest)[::-1][:num_features]

    # rearrange feature names so they match the sorted feature importances
    names_forest = [X.columns[i] for i in indices_forest]

    # create plot
    plt.figure(figsize=(num_features, num_features/1.5))
    plt.bar(range(num_features), imp_forest[indices_forest])

    # prettify plot
    plt.title('Feature Importance', fontsize=30, pad=15)
    plt.ylabel('Average Decrease in Impurity', fontsize=22, labelpad=20)
    # add feature names as x-axis labels
    plt.xticks(range(num_features), names_forest, fontsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)

    # Show plot
    plt.tight_layout()
    plt.show()
    
    if to_print:
        # print a list of feature names and their prevalance in the forest
        print([(i,j) for i,j in zip(names_forest, imp_forest[indices_forest])])
    
        
## The following code was copied from https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
## It has been edited slightly to suit my purposes.

def get_nearest(src_points, candidates, k_neighbors=3):
    """Find nearest neighbors for all source points from a set of candidate points"""
    # Note: haversine distance which is implemented here is a bit slower than using e.g. 'euclidean' metric
    # but useful as we get the distance between points in meters

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances
    # note: since I'm running this within the same dataframe, take index 1,
    # since index 0 will have a distance of 0, it being the same point.
    closest = indices[1]
    closest_dist = distances[1]

    # Return indices and distances
    return (closest, closest_dist)

def nearest_neighbor(gdf1, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    gdf2 = gdf1.copy().reset_index(drop=True)

    first_geom_col = gdf1.geometry.name
    second_geom_col = gdf2.geometry.name

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    first_radians = np.array(gdf1[first_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    second_radians = np.array(gdf2[second_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in gdf2 that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=first_radians, candidates=second_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = gdf2.loc[closest]

    # Ensure that the index corresponds to the one in gdf1
    closest_points = closest_points.reset_index(drop=True)
    
# the following commented code does not achieve desired result:    
#     conditions = [gdf1.iloc[closest].health == 'Poor', gdf1.iloc[closest].health == 'Fair']
#     choices = [0,1]

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius
# the following commented code does not achieve desired result:
#         closest_points['neighbor_health'] = np.select(conditions, choices, 2)

    return closest_points