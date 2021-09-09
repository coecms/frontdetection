import unittest
import numpy as np
import xarray as xr
from fronts import *

class DataGenerator:
    def get_latitude(self, nlat=None, latitudes=None):
        if latitudes is not None:
            nlat = len(latitudes)
        elif nlat is None:
            nlat = 21
            latitudes = np.linspace(-90, 90, nlat, endpoint=True)
        else:
            latitudes = np.linspace(-90, 90, nlat, endpoint=True)
        return xr.DataArray(
            latitudes, dims=('latitude',), coords={'latitude': latitudes},
            attrs={'units': 'degree_north', 'name': 'Latitude'}
        )        

    def get_longitude(self, nlon=None, longitudes=None):
        if longitudes is not None:
            nlon = len(longitudes)
        elif nlon is None:
            nlon = 40
            longitudes = np.linspace(0, 360, nlon, endpoint=False)
        else:
            longitudes = np.linspace(0, 360, nlon, endpoint=False)
        return xr.DataArray(
            longitudes, dims=('longitude',), coords={'longitude': longitudes},
            attrs={'units': 'degree_east', 'name': 'Longitude'}
        )        

    def get_temp(self, data=None, units="K"):
        if data is None:
            nlat = 21
            nlon = 40
            data = np.ones(nlat, nlon, dtype=np.float32)*285.0
        else:
            nlat = len(data)
            nlon = len(data[0])
            data = np.ndarray(data)
        lat = self.get_latitude(nlat=nlat)
        lon = self.get_longitude(nlon=nlon)
        return xr.DataArray(
            data, dims=('latitude', 'longitude'), 
            coords={'latitude': lat, 'longitude': lon},
            attrs={'name': 'temperature', 'units': units}
        )


class TestTest(unittest.TestCase):
    def test_saturation_pressure_negative(self):
        t_in = -12.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 2.4459, places=4)
    
    def test_saturation_pressure_zero(self):
        t_in = 0.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 6.1094, places=4)

    def test_saturation_pressure_positive(self):
        t_in = 21.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 24.81888, places=4)


if __name__ == '__main__':
    unittest.main()