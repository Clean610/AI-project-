import openpyxl
import pandas as pd
import numpy as np

array_1 = np.load('test_features.npy')

# print(array_1)
# print(type(array_1))
df = pd.DataFrame (array_1)



filepath = 'test_watch.xlsx'

df.to_excel(filepath, index=True)