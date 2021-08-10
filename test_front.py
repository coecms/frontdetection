import fronts
import json
import xarray as xr
import numpy as np

def test_era5():

    tafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/t/2020/t_era5_oper_pl_20200101-20200131.nc')
    uafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/u/2020/u_era5_oper_pl_20200101-20200131.nc')
    vafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/v/2020/v_era5_oper_pl_20200101-20200131.nc')
    husfile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/q/2020/q_era5_oper_pl_20200101-20200131.nc')

    n = 0

    # 32 index in level --> 900 hPa
    ta=tafile[n,32,0:100,0:100]
    ua=uafile[n,32,0:100,0:100]
    va=vafile[n,32,0:100,0:100]
    hus=husfile[n,32,0:100,0:100]
    
    t_wet=fronts.wetbulb(ta,hus,900,steps=120)
    
    frontdata=fronts.front(t_wet,ua,va,threshold_i=-1e-10,numsmooth=9,minlength=50)
    
    timestring=np.datetime_as_string(tafile.time.data[n],unit='h')
    with open(f'tests/900hPa_fronts_{timestring}.json') as sample_file:
        sample = json.load(sample_file)

        assert frontdata == sample
