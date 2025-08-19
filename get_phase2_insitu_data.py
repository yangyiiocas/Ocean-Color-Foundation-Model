# coding: utf-8
import os, glob, re, calendar, json, h5py, pickle
import argparse
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import scipy.io as scio
import sdata

class tasks2Data():
    def __init__(self, name):
        self.name = name
        self.targ = []
        self.sate = []
        self.date = []
        self.lat = []
        self.lon = []
        self.depth = []
        self.length = 0


def clean(lat,lon,dates, targ):
    m = len(lat)
    cond = {}
    idx = np.zeros(m,dtype=bool)
    for i in range(m):
        a = f'{round(lat[i]*24):0>5}&{round(lon[i]*24):0>5}&{dates[i].strftime("%Y%m%d")}'
        if a not in cond:
            cond[a] = []
            idx[i] = True
        cond[a].append(targ[i])
    return idx, np.array([np.nanmean(cond[a]) for a in cond])    

def get_tasks2(use_varis=None):
    tasks2 = {}

    # load data from txt
    for file in glob.glob('../采样dataset/*.txt'):
        p = re.findall(r'\[([^]]+)\]', file)[0]
        dat = pd.read_csv(file, delimiter='\t')
        if use_varis is not None:
            if p not in use_varis:
                continue
        print('Read:  ', file)
        if p not in tasks2:
            tasks2[p] = tasks2Data(p)
        tasks2[p].targ.extend(dat[dat.columns[-1]])
        tasks2[p].date.extend([datetime.fromisoformat(t) for t in dat['date time']])
        tasks2[p].lat.extend(dat['lat'])
        tasks2[p].lon.extend(dat['lon'])
        tasks2[p].depth.extend(dat['depth (m)'])
        tasks2[p].length +=dat[dat.columns[-1]].shape[0]
        with h5py.File(file.replace('采样dataset','遥感sat_matchups').replace('.txt','.mat'), 'r') as f:
            sate = f['sate'][:].transpose(2,1,0)
        # sate = scio.loadmat(file.replace('海洋水色dataset','sat_matchups').replace('.txt','.mat'))['sate']
        tasks2[p].sate.append(sate)
    for p in tasks2:
        tasks2[p].targ = np.array(tasks2[p].targ)
        tasks2[p].date = np.array(tasks2[p].date)
        tasks2[p].lat = np.array(tasks2[p].lat)
        tasks2[p].lon = np.array(tasks2[p].lon)
        tasks2[p].depth = np.abs(np.array(tasks2[p].depth))
        sate = np.concatenate(tasks2[p].sate, axis=0)
        a = sate[:,:10,:]
        a[a<=0] = np.nan
        sate[:,:10,:] = np.log10(a +1e-5)
        tasks2[p].sate = sate
    # clean data
    for p in tasks2:
        print(f'{p:15} total dataset: {tasks2[p].length:5}',end=';\t')
        tasks2[p].targ[tasks2[p].depth>=30] = np.nan
        tasks2[p].targ[np.abs(tasks2[p].targ)>=1e7] = np.nan
        targ = tasks2[p].targ
        v_995_max = np.percentile(targ[~np.isnan(targ)], 99.99)
        tasks2[p].targ[targ>v_995_max] = np.nan
        sate_nan = np.isnan(tasks2[p].sate).sum(axis=2).max(axis=1)>=30
        tasks2[p].targ[sate_nan] = np.nan
        

        cho_idx = (~np.isnan(tasks2[p].targ))&(tasks2[p].targ>0)
        tasks2[p].targ = tasks2[p].targ[cho_idx]
        tasks2[p].lat = tasks2[p].lat[cho_idx]
        tasks2[p].lon = tasks2[p].lon[cho_idx]
        tasks2[p].depth = tasks2[p].depth[cho_idx]
        tasks2[p].date = tasks2[p].date[cho_idx]
        tasks2[p].sate = tasks2[p].sate[cho_idx]
        tasks2[p].length = tasks2[p].targ.shape[0]
        
        [cho_idx, targ] = clean(tasks2[p].lat,tasks2[p].lon,tasks2[p].date,tasks2[p].targ)
        tasks2[p].targ = targ
        tasks2[p].lat = tasks2[p].lat[cho_idx]
        tasks2[p].lon = tasks2[p].lon[cho_idx]
        tasks2[p].depth = tasks2[p].depth[cho_idx]
        tasks2[p].date = tasks2[p].date[cho_idx]
        tasks2[p].sate = tasks2[p].sate[cho_idx]
        tasks2[p].length = tasks2[p].targ.shape[0]
        
        print(f'train: {tasks2[p].targ.shape[0]:5}')

    for vari in tasks2:        
        seq = tasks2[vari].sate.shape[2]
        lat_g = np.tile(tasks2[vari].lat.reshape(-1, 1, 1), (1,1,seq))
        lon_g = np.tile(tasks2[vari].lon.reshape(-1, 1, 1), (1,1,seq))
        ymds, mth = [], []
        for dt in tasks2[vari].date:
            ymds.append(np.array([t.timetuple().tm_yday / (366. if calendar.isleap(t.year) else 365.) for t in [dt-timedelta(days=i) for i in range(seq)]], dtype=np.float32))
            mth.append(np.array([t.month / 12. for t in [dt-timedelta(days=i) for i in range(seq)]],dtype=np.float32))
        day_g = np.stack(ymds, axis=0)[:,np.newaxis,:]
        mon_g = np.stack(mth, axis=0)[:,np.newaxis,:]

        tasks2[vari].sate = np.concatenate([tasks2[vari].sate,
                                            np.cos(lon_g/180.*np.pi),
                                            np.sin(lon_g/180.*np.pi),
                                            lat_g/90.,
                                            np.sin(day_g*2*np.pi),
                                            np.cos(day_g*2*np.pi),
                                            np.sin(mon_g*2*np.pi),
                                            np.cos(mon_g*2*np.pi)], axis=1)

    for vari in tasks2:
        tasks2[vari].targ = tasks2[vari].targ if vari in sdata.no_log10 else np.log10(tasks2[vari].targ)
    
    return tasks2

if __name__ == '__main__':
    tasks2 = get_tasks2()
    with open('phase2_insitu_data.pkl','wb') as f:
        pickle.dump(tasks2, f)
