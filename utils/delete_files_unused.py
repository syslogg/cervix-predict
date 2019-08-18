# This file is utilities to delete files always preprocessed from Herlev Dataset

import os

folders = []

pathHerlev = '.\\dataset\\herlev\\'

for r, d, f in os.walk(pathHerlev):
    for folder in d:
        fieldPath = os.path.join(r, folder)
        for files in os.listdir(fieldPath):
                if '-d.bmp' in files:
                        # Remove Files
                        fileToRemove = os.path.join(fieldPath, files)
                        if os.path.exists(fileToRemove):
                                os.remove(fileToRemove)
            
