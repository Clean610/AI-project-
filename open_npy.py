import openpyxl
import pandas as pd
import numpy as np

array_1 = np.load('./ml_speech_projects/mfcc13_models_2021y11m16d17h8m56s/data_train/train_features.npy')

# print(array_1)
# print(type(array_1))
df = pd.DataFrame (array_1)



filepath = 'train_watch.xlsx'

df.to_excel(filepath, index=True)