import pickle, glob, h5py
import numpy as np

from get_phase2_insitu_data import get_tasks2
from get_phase2_insitu_data import tasks2Data
import config 


sate = []
for file in glob.glob('../遥感sat_matchups/*.mat'):
    with h5py.File(file, 'r') as f:
        data = f['sate'][:].transpose(2,1,0)
    print(file)
    print(data.shape)
    sate.append(data)
sate = np.concatenate(sate, axis=0)

sate_nan = np.isnan(sate).sum(axis=2).max(axis=1)>=30
sate = sate[~sate_nan]
sate = np.isnan(sate)
with open('mask_extra.pkl', 'wb') as f:
    pickle.dump(sate, f)

print('合并，shape: ', sate.shape)
