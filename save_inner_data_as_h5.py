# coding:utf-8
import os,glob,time,calendar,timeit, h5py
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import torch.utils.data
import concurrent.futures
import multiprocessing
from dataset_psave import read_nc

def save_inner_data(args):
    date, rpath, to_log10, inpt_varis, targ_varis, save_rpath, seq = args

    # targets
    targs = {vari:read_nc((date,rpath+sub, to_log10[vari])) for vari,sub in targ_varis.items()}
    no_nan = np.ones((4320, 8640), dtype=bool)
    for _, values in targs.items():
        no_nan = no_nan&np.isfinite(values)
    for vari, values in targs.items():
        targs[vari] = values[no_nan]

    # inputs
    inpt = []
    for t in range(seq):
        s1 = [read_nc((date-timedelta(days=t),rpath+sub, to_log10[vari]))[no_nan] for vari,sub in inpt_varis.items()]
        inpt.append(np.stack(s1,axis=1))
    inpt = np.stack(inpt,axis=2)

    dx = 1/24
    lat = np.arange(90-dx/2,-90-dx/2,-dx, dtype=np.float16)
    lon = np.arange(-180+dx/2, 180+dx/2,dx, dtype=np.float16)
    dts = [date-timedelta(days=t) for t in range(seq)]
    lat_g, lon_g = np.meshgrid(lat,lon)
    lat_g = np.tile(lat_g.T[no_nan].reshape(-1,1,1), (1, 1, seq))
    lon_g = np.tile(lon_g.T[no_nan].reshape(-1,1,1), (1, 1, seq))

    ymd = np.array([t.timetuple().tm_yday / (366 if calendar.isleap(t.year) else 365) for t in dts], dtype=np.float16)
    mth = np.array([t.month / 12. for t in dts],dtype=np.float16)
    day_g = np.tile(ymd.reshape(1, 1, -1), (inpt.shape[0], 1, 1))
    mon_g = np.tile(mth.reshape(1, 1, -1), (inpt.shape[0], 1, 1))

    inpt = np.concatenate([inpt,
                           np.cos(lon_g/180.*np.pi),
                           np.sin(lon_g/180.*np.pi),
                           lat_g/90.,
                           np.sin(day_g*2*np.pi),
                           np.cos(day_g*2*np.pi),
                           np.sin(mon_g*2*np.pi),
                           np.cos(mon_g*2*np.pi)], axis=1)
    

    cho_idx = np.ones(inpt.shape[0], dtype=bool)
    inpt_isnan = np.isnan(inpt)|np.isinf(inpt)
    inpt_isnan = inpt_isnan.sum(axis=2).max(axis=1)>=26.
    cho_idx[inpt_isnan] = False
    inpt = inpt[cho_idx]
    targs = {vari:values[cho_idx] for vari, values in targs.items()}

    print(save_rpath+date.strftime("%Y-%m-%d")+'done!')
    with h5py.File(save_rpath+date.strftime("%Y-%m-%d")+'.h5', 'w') as hf:
        hf.create_dataset("inpt", data=inpt, compression="gzip")
        for vari, values in targs.items():
            hf.create_dataset(vari, data=values, compression="gzip")





if __name__ == '__main__':
    # tasks brfore none 
    from sdata import to_log10, tasks, inpt_varis
    rpath = '/mnt/Aqua-MODIS/'
    save_rpath = '/mnt/oc_temp/'
    args_list = []
    for date in pd.date_range("2015-12-12","2015-12-31"):
        if not os.path.exists(save_rpath+date.strftime("%Y-%m-%d")+'.h5'):
            print(date)
            args_list.append((date, rpath, to_log10, inpt_varis, tasks, save_rpath, 30))
    with multiprocessing.Pool(4) as pool:
        processed_chunks = pool.map(save_inner_data, args_list)
