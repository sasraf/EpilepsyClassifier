from glob import glob
import numpy as np
import pickle

data = list()
outputs = list()

# Saves all top level dirs in data
dataFolders = glob('data/*')

# dict containing info on which folders are healthy, ictal
folderKeys = {}
folderKeys['Z'] = 'healthy'
folderKeys['O'] = 'healthy'
folderKeys['F'] = 'epileptic'
folderKeys['N'] = 'epileptic'
folderKeys['S'] = 'seizure'

# For each folder, for each text file in folder:
# append text file contents to data in numpy array, append output depending on if text file
# contains ictal activity or not
for folder in dataFolders:
    health = folderKeys[folder[len(folder) - 1]]

    print("Current folder: " + folder)

    fileLoc = glob(folder + '\\*')

    for file in fileLoc:
        array = np.loadtxt(file)

        data.append(array)

        if health is 'healthy':
            outputs.append(np.array((1, 0)))

        else:
            outputs.append(np.array((0, 1)))

# Store in dict for ease of serialization
seizureDict = dict()
seizureDict['inputs'] = data
seizureDict['outputs'] = outputs

# Dump data
pickle.dump(seizureDict, open('seizureDataSerialized.txt', 'wb'))

print("dumped")
