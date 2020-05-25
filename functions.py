import pandas as pd
import numpy as np

def yes_to_one(df, cols):
    '''Turn columns with 'Yes' and 'No' values into 1s and 0s.
    Overwrites the input columns!
    '''
    for col in cols:
        df[col] = np.where(df[col] == 'Yes', 1, 0)