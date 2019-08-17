import os

folders = []

pathHerlev = '.\\dataset\\herlev\\'
for r, d, f in os.walk(pathHerlev):
    for folder in d:
        fieldPath = os.path.join(r, folder)
        folders.append(os.path.join(r, folder))
        print(os.listdir(fieldPath))


for f in folders:
    print(f)