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

def test_dewpoint():
    """
    Test the metpy and internal dewpoint routines are consistent
    """

    test_data = xr.open_dataset('tests/front_test.nc')

    dp = dewpoint_from_specific_humidity(test_data.level, test_data.t, test_data.q).metpy.convert_units('degK')
    dp2 = fronts.dewpoint(test_data.t, test_data.q, test_data.level, ta_units=str(test_data.t.metpy.units)) * units.degK

    assert_allclose(dp, dp2)

def test_wetbulb():
    """
    Test the metpy and internal wetbulb routines are consistent
    """

    test_data = xr.open_dataset('tests/front_test.nc').metpy.quantify()

    dp = dewpoint_from_specific_humidity(test_data.level, test_data.t, test_data.q).metpy.convert_units('degK')
    dp2 = wet_bulb_temperature(test_data.level, test_data.t, dp) * units.degK

    assert_allclose(dp, dp2)

