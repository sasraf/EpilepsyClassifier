from glob import glob
import numpy as np
# import mne

healthyInputData = list()
ictalInputData = list()

# Saves all top level dirs in data
dataFolders = glob('data/*')

folderKeys = {}
folderKeys['Z'] = 'healthy'
folderKeys['O'] = 'healthy'
folderKeys['F'] = 'epileptic'
folderKeys['N'] = 'epileptic'
folderKeys['S'] = 'seizure'

for folder in dataFolders:
    health = folderKeys[folder[len(folder) - 1]]

    fileLoc = glob(folder + '\\*')

    for file in fileLoc:
        array = np.loadtxt(file)

        if health is 'healthy':
            healthyInputData.append(array)

        else:
            ictalInputData.append(array)




# print(dataFolders)
