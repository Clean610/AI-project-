from numpy.core.defchararray import index
from openpyxl import load_workbook
import numpy as np
import pandas as pd

df = pd.read_excel('y_pred12.xlsx')

filepath = 'y11.npy'


df.to_records(filepath), index=True)