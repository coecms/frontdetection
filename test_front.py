import fronts
import json
import xarray as xr
import numpy as np
from xarray.testing import assert_allclose

from metpy.calc import wet_bulb_temperature, dewpoint_from_specific_humidity
from metpy.units import units

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

def test_metpy():
    """
    Test the metpy and internal routines are consistent
    """

    test_data = xr.open_dataset('tests/front_test.nc')

    dewpoint = dewpoint_from_specific_humidity(test_data.level, 
                                         test_data.t, 
                                         test_data.q).metpy.convert_units('degK')
    dewpoint2 = fronts.dewpoint(test_data.t, 
                                test_data.q, 
                                test_data.level, 
                                ta_units=str(test_data.t.metpy.units)) * units.degK

    assert_allclose(dewpoint, dewpoint2)

    # Uses the dewpoint calculated above
    wb_temp = wet_bulb_temperature(test_data.level, 
                                  test_data.t, 
                                  dewpoint).metpy.convert_units('degK')
    wb_temp2 = fronts.wetbulb(test_data.t, 
                              test_data.q, 
                              test_data.level, 
                              steps=100, 
                              ta_units=str(test_data.t.metpy.units)).metpy.convert_units('degK')

    # This result is not as close as dewpoint. Still probably good enough
    assert_allclose(wb_temp, wb_temp2, rtol=5e-3)

