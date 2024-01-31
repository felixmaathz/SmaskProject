import pandas as pd 
import numpy as np
from scipy import stats 

data = pd.read_csv('siren_data_train.csv')

distances = {'distance': []}
for i in range(0, len(data)):
    point_a = np.array((float(data.iloc[i, 1]), float(data.iloc[i, 2])))
    point_b = np.array((float(data.iloc[i, 6]), float(data.iloc[i, 7])))
    distance = np.linalg.norm(point_a - point_b)
    distances['distance'].append(distance)


data = data.assign(distance=distances['distance'])

data = data.drop(['near_x', 'near_y', 'ycoor', 'xcoor', 'near_fid', 'near_angle'], axis=1)

z = np.abs(stats.zscore(data['distance']))
threshold = 2
outlier_indices = np.where(z > threshold)[0]
data_no_outliers = data.drop(outlier_indices, axis=0)
data_no_outliers.dropna(inplace=True)

data_no_outliers.to_csv('siren_data_train_no_outliers.csv', index=False)