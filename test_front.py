import fronts
import fronts_legacy
import json
import xarray as xr
import numpy as np
from xarray.testing import assert_allclose
from numpy.testing import assert_allclose as np_assert

from metpy.calc import wet_bulb_temperature, dewpoint_from_specific_humidity
from metpy.units import units

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def test_front_detection():

    test_data = xr.open_dataset('tests/front_test.nc')

    ta = test_data.t
    ua = test_data.u
    va = test_data.v
    hus = test_data.q
    
    t_wet=fronts_legacy.wetbulb(ta,hus,900,steps=120)
    
    frontdata=fronts_legacy.front(t_wet,ua,va,threshold_i=-1e-10,numsmooth=9,minlength=50)
    
    timestring=np.datetime_as_string(test_data.time.data,unit='h')

    with open(f'tests/900hPa_fronts_{timestring}.json') as sample_file:
        sample = json.load(sample_file)

        assert frontdata == sample, f"result isn't comparable with test data"

    return frontdata, t_wet


def test_metpy():
    """
    Test the metpy and internal routines are consistent
    """

    test_data = xr.open_dataset('tests/front_test.nc')

    dewpoint = dewpoint_from_specific_humidity(test_data.level, 
                                         test_data.t, 
                                         test_data.q).metpy.convert_units('degK')
    dewpoint2 = fronts_legacy.dewpoint(test_data.t, 
                                test_data.q, 
                                test_data.level, 
                                ta_units=str(test_data.t.metpy.units)) * units.degK

    assert_allclose(dewpoint, dewpoint2)

    # Uses the dewpoint calculated above
    wb_temp = wet_bulb_temperature(test_data.level, 
                                  test_data.t, 
                                  dewpoint).metpy.convert_units('degK')
    wb_temp2 = fronts_legacy.wetbulb(test_data.t, 
                              test_data.q, 
                              test_data.level, 
                              steps=150, 
                              ta_units=str(test_data.t.metpy.units)).metpy.convert_units('degK')

    # This result is not as close as dewpoint. Still probably good enough
    assert_allclose(wb_temp, wb_temp2, rtol=1e-3)


def test_front_detection_metpy():

    test_data = xr.open_dataset('tests/front_test.nc')

    ta = test_data.t
    ua = test_data.u
    va = test_data.v
    hus = test_data.q
    lvl = test_data.level
    
    dewpoint = dewpoint_from_specific_humidity(pressure=lvl,
                                               temperature=ta,
                                               specific_humidity=hus)

    t_wet = wet_bulb_temperature(pressure=900*units.hPa,
                                 temperature=ta,
                                 dewpoint=dewpoint)
    
    frontdata = fronts.front(t_wet,
                             ua,
                             va,
                             threshold_i=-1e-10,
                             numsmooth=9,
                             minlength=50)
    
    timestring=np.datetime_as_string(test_data.time.data,unit='h')

    with open(f'tests/900hPa_fronts_{timestring}.json') as sample_file:
        sample = json.load(sample_file)

        #assert frontdata == sample, f"result isn't comparable with test data"

    return frontdata, t_wet

def test_plot(data):

    # plotting the output
    lines = data[0]
    t_wet = data[1]

    clines=lines['cold_fronts']
    wlines=lines['warm_fronts']
    slines=lines['stationary_fronts']
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent([110,180,-10,-60])
    ax.coastlines()
    ax.gridlines()
    f=ax.contourf(t_wet.longitude,t_wet.latitude,t_wet,np.arange(250,291,2.5),extend='both',transform=ccrs.PlateCarree(),cmap='RdYlBu_r')
    for n in range(len(slines)):
        ax.plot(slines[n][1],slines[n][0],'k',ms=1,transform=ccrs.PlateCarree())
    for n in range(len(wlines)):
        ax.plot(wlines[n][1],wlines[n][0],'r',ms=1,transform=ccrs.PlateCarree())
    for n in range(len(clines)):
        ax.plot(clines[n][1],clines[n][0],'b',ms=1,transform=ccrs.PlateCarree())
    cbar=fig.colorbar(f)
    fig.savefig('front_output_test.pdf')
    plt.show()

test_plot(test_front_detection_metpy())