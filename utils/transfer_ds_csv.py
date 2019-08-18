# This script can be used to create a csv file from dataset herlev

import os
import pandas as pd

path = '.\\dataset\\herlev\\'
data = []
# dsCsv = pd.DataFrame(columns=['file_path','diagnostic'])

for r, d, f in os.walk(path):
    for folder in d:
        fieldPath = os.path.join(r, folder)
        for files in os.listdir(fieldPath):
            data.append([os.path.join(fieldPath,files), folder])

df = pd.DataFrame(columns=['file_path', 'diagnostic'], data=data)
df.to_csv(r'ds.csv',index=False)