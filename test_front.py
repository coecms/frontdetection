import fronts
import json
import xarray as xr
import numpy as np

def test_front_detection():

    test_data = xr.open_dataset('tests/front_test.nc')

    ta = test_data.t
    ua = test_data.u
    va = test_data.v
    hus = test_data.q
    
    t_wet=fronts.wetbulb(ta,hus,900,steps=120)
    
    frontdata=fronts.front(t_wet,ua,va,threshold_i=-1e-10,numsmooth=9,minlength=50)
    
    timestring=np.datetime_as_string(test_data.time.data,unit='h')

    with open(f'tests/900hPa_fronts_{timestring}.json') as sample_file:
        sample = json.load(sample_file)

        assert frontdata == sample
