import numpy as np


full_name = {
    'chlor_a': 'Chlorophyll concentration',
    'nflh': 'Fluorescence Line Height (normalized)',
    'ipar': 'Instantaneous Photosynthetically Available Radiation',
    'adg_unc_443': 'Absorption due to gelbstoff and detritus Uncertainty at 443',
    'adg_443': 'Absorption due to gelbstoff and detritus at 443',
    'adg_s': 'Absorption due to gelbstoff and detritus slope parameter',
    'aph_unc_443': 'Absorption due to phytoplankton at uncertainty at 443 nm',
    'aph_443': 'Absorption due to phytoplankton at 443 nm',
    'bbp_unc_443': 'Particulate backscatter Uncertainty at 443 nm',
    'bbp_443': 'Particulate backscatter at 443 nm',
    'bbp_s': 'Particulate backscatter spectral slope parameter',
    'a_412': 'Total absorption at 412 nm',
    'a_443': 'Total absorption at 443 nm',
    'a_469': 'Total absorption at 469 nm',
    'a_488': 'Total absorption at 488 nm',
    'a_531': 'Total absorption at 531 nm',
    'a_547': 'Total absorption at 547 nm',
    'a_555': 'Total absorption at 555 nm',
    'a_645': 'Total absorption at 645 nm',
    'a_667': 'Total absorption at 667 nm',
    'a_678': 'Total absorption at 678 nm',
    'bb_412': 'Total backscattering at 412 nm',
    'bb_443': 'Total backscattering at 443 nm',
    'bb_469': 'Total backscattering at 469 nm',
    'bb_488': 'Total backscattering at 488 nm',
    'bb_531': 'Total backscattering at 531 nm',
    'bb_547': 'Total backscattering at 547 nm',
    'bb_555': 'Total backscattering at 555 nm',
    'bb_645': 'Total backscattering at 645 nm',
    'bb_667': 'Total backscattering at 667 nm',
    'bb_678': 'Total backscattering at 678 nm',
    'Kd_490': 'Diffuse attenuation coefficient at 490 nm',
    'sst_n': 'Sea Surface Temperature (11 u nighttime)',
    'par': 'Photosynthetically Available Radiation',
    'pic': 'Particulate Inorganic Carbon',
    'poc': 'Particulate Organic Carbon',
    'aot_869': 'Aerosol optical thickness at 869 nm',
    'angstrom': 'Angstrom coefficient',
    'Rrs_412': 'Remote sensing reflectance at 412 nm',
    'Rrs_443': 'Remote sensing reflectance at 443 nm',
    'Rrs_469': 'Remote sensing reflectance at 469 nm',
    'Rrs_488': 'Remote sensing reflectance at 488 nm',
    'Rrs_531': 'Remote sensing reflectance at 531 nm',
    'Rrs_547': 'Remote sensing reflectance at 547 nm',
    'Rrs_555': 'Remote sensing reflectance at 555 nm',
    'Rrs_645': 'Remote sensing reflectance at 645 nm',
    'Rrs_667': 'Remote sensing reflectance at 667 nm',
    'Rrs_678': 'Remote sensing reflectance at 678 nm',
    'sst': 'Sea Surface Temperature (11 u daytime)',
    'sst4_n': 'Sea Surface Temperature (4 u nighttime)'
}

    
to_log10 = {
    'chlor_a': True,
    'nflh': True,
    'ipar': False,
    'adg_unc_443': True,
    'adg_443': True,
    'adg_s': False,
    'aph_unc_443': True,
    'aph_443': True,
    'bbp_unc_443': True,
    'bbp_443': True,
    'bbp_s': False,
    'a_412': True,
    'a_443': True,
    'a_469': True,
    'a_488': True,
    'a_531': True,
    'a_547': True,
    'a_555': True,
    'a_645': True,
    'a_667': True,
    'a_678': True,
    'bb_412': True,
    'bb_443': True,
    'bb_469': True,
    'bb_488': True,
    'bb_531': True,
    'bb_547': True,
    'bb_555': True,
    'bb_645': True,
    'bb_667': True,
    'bb_678': True,
    'Kd_490': True,
    'pic': True,
    'poc': True,
    'aot_869': True,
    'angstrom': True,
    'Rrs_412': True,
    'Rrs_443': True,
    'Rrs_469': True,
    'Rrs_488': True,
    'Rrs_531': True,
    'Rrs_547': True,
    'Rrs_555': True,
    'Rrs_645': True,
    'Rrs_667': True,
    'Rrs_678': True,
    'par': False,
    'sst': False,
    'sst_n': False,
    'sst4_n': False,
}

no_log10 = ['Fm','Fn','Fp','pp']

tasks = {
    'chlor_a': full_name['chlor_a'],
    # 'nflh': full_name['nflh'],
    # 'ipar': full_name['ipar'],
    # 'adg_unc_443': full_name['adg_unc_443'],
    'adg_443': full_name['adg_443'],
    # 'adg_s': full_name['adg_s'],
    # 'aph_unc_443': full_name['aph_unc_443'],
    'aph_443': full_name['aph_443'],
    # 'bbp_unc_443': full_name['bbp_unc_443'],
    'bbp_443': full_name['bbp_443'],
    # 'bbp_s': full_name['bbp_s'],
    'a_412': full_name['a_412'],
    'a_443': full_name['a_443'],
    'a_469': full_name['a_469'],
    'a_488': full_name['a_488'],
    'a_531': full_name['a_531'],
    'a_547': full_name['a_547'],
    'a_555': full_name['a_555'],
    'a_645': full_name['a_645'],
    'a_667': full_name['a_667'],
    'a_678': full_name['a_678'],
    'bb_412': full_name['bb_412'],
    'bb_443': full_name['bb_443'],
    'bb_469': full_name['bb_469'],
    'bb_488': full_name['bb_488'],
    'bb_531': full_name['bb_531'],
    'bb_547': full_name['bb_547'],
    'bb_555': full_name['bb_555'],
    'bb_645': full_name['bb_645'],
    'bb_667': full_name['bb_667'],
    'bb_678': full_name['bb_678'],
    'Kd_490': full_name['Kd_490'],
    'pic': full_name['pic'],
    'poc': full_name['poc'],
}



inpt_varis = {
    'Rrs_412': full_name['Rrs_412'],
    'Rrs_443': full_name['Rrs_443'],
    'Rrs_469': full_name['Rrs_469'],
    'Rrs_488': full_name['Rrs_488'],
    'Rrs_531': full_name['Rrs_531'],
    'Rrs_547': full_name['Rrs_547'],
    'Rrs_555': full_name['Rrs_555'],
    'Rrs_645': full_name['Rrs_645'],
    'Rrs_667': full_name['Rrs_667'],
    'Rrs_678': full_name['Rrs_678'],
    'par': full_name['par'],
    'sst': full_name['sst'],
    'sst_n': full_name['sst_n'],
    'sst4_n': full_name['sst4_n'],
}

def tasks_extra():
    tasks = {'Fm':'',
             'Fn':'',
             'Fp':'',
             'Cm':'',
             'Cn':'',
             'Cp':''}
    bands = ['412','443','469','488','531','547','555','645','667','678']
    for i,band1 in enumerate(bands):
            for j,band2 in enumerate(bands):
                if (i!=j):
                    vari = 'Index-' + band1 + '_' + band2
                    tasks[vari] = 'Index, band1:{band1} band2:{band2}'
                    vari = 'Ratio-' + band1 + '_' + band2
                    tasks[vari] = 'Ratio, log10, band1:{band1} band2:{band2}'
    return tasks
# add extra tasks
def tasks_extra_fun(inpt, targs):
    plus_targs = {}
    Rrs = 10**inpt[:,:10,0].copy()
    bands = ['412','443','469','488','531','547','555','645','667','678']
    Rrs_bands = {band:Rrs[:,i] for i,band in enumerate(bands)}
    
    for i,band1 in enumerate(bands):
        for j,band2 in enumerate(bands):
            vari = 'Index-' + band1 + '_' + band2
            if (i!=j)&(vari not in plus_targs):
                plus_targs[vari] = (Rrs[:,i] - Rrs[:,j]) / (Rrs[:,i] + Rrs[:,j])

    for i,band1 in enumerate(bands):
        for j,band2 in enumerate(bands):
            vari = 'Ratio-' + band1 + '_' + band2
            if (i!=j)&(vari not in plus_targs):
                tmp = np.log10(Rrs[:,j] / Rrs[:,i])
                tmp[np.abs(tmp)>6] = np.nan
                plus_targs[vari] = tmp



    tchla = 10**(targs['chlor_a'].copy())
    Cm_pn, Cm_p, D_pn, D_p = 0.77, 0.13, 0.94, 0.80
    Cp = Cm_p*(1-np.exp(-D_p/Cm_p*tchla))
    Cn = Cm_pn*(1-np.exp(-D_pn/Cm_pn*tchla))-Cp
    Cm = tchla-Cm_pn*(1-np.exp(-D_pn/Cm_pn*tchla))
    plus_targs['Fp'] = Cp/(tchla +1e-4)
    plus_targs['Fn'] = Cn/(tchla +1e-4)
    plus_targs['Fm'] = Cm/(tchla +1e-4)
    plus_targs['Cp'] = np.log10(Cp +1e-4)
    plus_targs['Cn'] = np.log10(Cn +1e-4)
    plus_targs['Cm'] = np.log10(Cm +1e-4)
    

    return plus_targs