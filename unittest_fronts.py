import unittest
import numpy as np
from scipy.signal.ltisys import ZerosPolesGain
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


class Test_Saturation_Pressure(unittest.TestCase):
    def test_saturation_pressure_negative(self):
        t_in = -12.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 2.4459, places=4)
    
    def test_saturation_pressure_zero(self):
        t_in = 0.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 6.1094, places=4)

    def test_saturation_pressure_positive(self):
        t_in = 21.0 + 273.15
        self.assertAlmostEqual(calculate_saturation_pressure(t_in), 24.81888, places=4)


# class Test_Dewpoint_Temperature(unittest.TestCase):
#     def test_dewpoint_temperature(self):
#         self.assertFalse(True, msg="Dewpoint Temperature Tests not yet implemented")


# class Test_Relative_Humidity(unittest.TestCase):
#     def test_relative_humidity(self):
#         self.assertFalse(True, msg="Relative Humidity Test not yet implemented")

class Test_Zeropoints(unittest.TestCase):

    def create_data(self, data):
        d = np.array(data)
        d1 = np.arange(d.shape[0])
        d2 = np.arange(d.shape[1])
        return d, d1, d2

    def test_no_zeropoints(self):
        d, d1, d2 = self.create_data(
            [[1, 1, 1], 
             [1, 1, 1],
             [1, 1, 1]])
        self.assertEqual(zeropoints(d, d1, d2).size, 0)

    def test_zeropoints_on_grid(self):
        d, d1, d2 = self.create_data(
            [[-1, 0, 1], 
             [-1, np.nan, 1],
             [-1, np.nan, 1]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[0, 1]])
        np.testing.assert_allclose(res, expected)

    def test_zeropoints_between_grid(self):
        d, d1, d2 = self.create_data(
            [[-1, 1, 1], 
             [-1, 1, 1],
             [-1, 1, 1]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[0, 0.5], [1, 0.5], [2, 0.5]])
        np.testing.assert_allclose(res, expected)

    def test_zeropoints_on_grid_y(self):
        d, d1, d2 = self.create_data(
            [[-1, -1, -1], 
             [ 0,  0,  0],
             [ 1,  1,  1]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[1, 0], [1, 1], [1, 2]])
        np.testing.assert_allclose(res, expected)

    def test_zeropoints_between_grid_y(self):
        d, d1, d2 = self.create_data(
            [[-1, -1, -1], 
             [1, 1, 1],
             [1, 1, 1]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[0.5, 0], [0.5, 1], [0.5, 2]])
        np.testing.assert_allclose(res, expected)

    def test_zeropoints_fractional(self):
        d, d1, d2 = self.create_data(
            [[-1, -1, -1], 
             [3, 3, 3],
             [1, 1, 1]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[0.25, 0], [0.25, 1], [0.25, 2]])
        np.testing.assert_allclose(res, expected)

    def test_zeropoints_diagonal(self):
        d, d1, d2 = self.create_data(
            [[-3, -1, 2], 
             [-2, 0, 1],
             [1, 1, 4]]
        )
        res = zeropoints(d, d1, d2)
        expected = np.array([[1, 1], [0, 4/3], [5/3, 0]])
        np.testing.assert_allclose(res, expected)



if __name__ == '__main__':
    unittest.main()