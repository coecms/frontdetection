# Created 16/08/2022
# This script contains all of the removed functions from fronts.py if they are ever needed.

import numpy as np
import scipy.signal
import xarray as xr
import scipy.spatial.distance as sp_dist
import geopy.distance as gp_dist

# This function has been replaced by wet_bulb_temperature from metpy.
def wetbulb(ta, hus, plev, steps=100, ta_units=None):
    # calculates wetbulb temperature from pressure-level data
    # Inputs: ta - temperature field (xarray)
    #         hus - specific humidity field (xarray)
    #         plev - the level of the data (in hPa, scalar)
    #         steps -  the number of steps in the numerical calculation
    #         ta_units - the units of the temperature field (if not provided, read from ta)
    if ta_units == None:
        ta_units = ta.units
    # saturation vapor pressure
    if ta_units == "K" or ta_units == "Kelvin" or ta_units == "kelvin":
        es = 6.1094 * np.exp((17.625 * (ta - 273.15)) / (ta - 30.11))
    elif (
        ta_units == "C"
        or ta_units == "degC"
        or ta_units == "deg_C"
        or ta_units == "Celcius"
        or ta_units == "celcius"
    ):
        es = 6.1094 * np.exp((17.625 * (ta)) / (ta + 243.04))
    else:
        raise ValueError(
            "Input temperature unit not recognised, use Kelvin (K) or Celcius (C, degC, deg_C)"
        )
    # relative humidity from specific humidity and sat. vap. pres.
    rh = (hus * (plev - es)) / (0.622 * (1 - hus) * es)
    # vapor pressure
    e = es * rh
    # dewpoint temperature
    t_dewpoint = ((243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))) + 273.15

    # unlike the above, calculating the wetbulb temperature is done numerically
    delta_t = (ta - t_dewpoint) / steps
    cur_diff = np.abs(es - e)
    t_wet = ta.copy()

    for i in range(steps):
        cur_t = ta - i * delta_t
        es_cur_t = 6.1094 * np.exp((17.625 * (cur_t - 273.15)) / (cur_t - 30.11))
        adiabatic_adj = 850 * (ta - cur_t) * (0.00066) * (1 + 0.00115 * cur_t)
        diff = np.abs(es_cur_t - adiabatic_adj - e)
        t_wet.data[diff < cur_diff] = cur_t.data[diff < cur_diff]
        cur_diff.data[diff < cur_diff] = diff.data[diff < cur_diff]

    return t_wet

# This function has been replaced by dewpoint_from_specific_humidity from metpy.
def dewpoint(ta, hus, plev, ta_units=None):
    # calculates dewpoint temperature from pressure-level data
    # Inputs: ta - temperature field (xarray)
    #         hus - specific humidity field (xarray)
    #         plev - the level of the data (in hPa, scalar)
    #         ta_units - the units of the temperature field (if not provided, read from ta)
    if ta_units == None:
        ta_units = ta.units
    if ta_units == "K" or ta_units == "Kelvin" or ta_units == "kelvin":
        es = 6.1094 * np.exp((17.625 * (ta - 273.15)) / (ta - 30.11))
    elif (
        ta_units == "C"
        or ta_units == "degC"
        or ta_units == "deg_C"
        or ta_units == "Celcius"
        or ta_units == "celcius"
    ):
        es = 6.1094 * np.exp((17.625 * (ta)) / (ta + 243.04))
    else:
        raise ValueError(
            "Input temperature unit not recognised, use Kelvin (K) or Celcius (C, degC, deg_C)"
        )
    rh = (hus * (plev - es)) / (0.622 * (1 - hus) * es)
    e = es * rh
    t_dewpoint = ((243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))) + 273.15
    return t_dewpoint

def zeropoints(data, dim1, dim2):
    # finds zero-crossing points in a gridded data set along the lines of each dimension
    # inputs: data - 2d data field (numpy array)
    #         dim1 - coords of the first dim of data (np array)
    #         dim2 - coords of the second dim of data (np array)
    n1, n2 = data.shape
    # assuming regularly spaced grid:
    d_dim2 = dim2[1] - dim2[0]
    d_dim1 = dim1[1] - dim1[0]
    tloc_1 = []
    tloc_2 = []
    for lonn in range(0, n2 - 1):
        for latn in range(0, n1 - 1):
            flag = False
            if data[latn, lonn] == 0:
                tloc_1.append([dim1[latn], dim2[lonn]])
                flag = True
            else:
                if (
                    np.isfinite(data[latn, lonn])
                    and np.isfinite(data[latn, lonn + 1])
                    and not flag
                ):
                    if (data[latn, lonn] > 0 and data[latn, lonn + 1] < 0) or (
                        data[latn, lonn] < 0 and data[latn, lonn + 1] > 0
                    ):
                        tloc_1.append(
                            [
                                dim1[latn],
                                dim2[lonn]
                                + d_dim2
                                * np.abs(
                                    data[latn, lonn]
                                    / (data[latn, lonn] - data[latn, lonn + 1])
                                ),
                            ]
                        )
                if (
                    np.isfinite(data[latn, lonn])
                    and np.isfinite(data[latn + 1, lonn])
                    and not flag
                ):
                    if (data[latn, lonn] > 0 and data[latn + 1, lonn] < 0) or (
                        data[latn, lonn] < 0 and data[latn + 1, lonn] > 0
                    ):
                        tloc_2.append(
                            [
                                dim1[latn]
                                + d_dim1
                                * np.abs(
                                    data[latn, lonn]
                                    / (data[latn, lonn] - data[latn + 1, lonn])
                                ),
                                dim2[lonn],
                            ]
                        )
    return np.array(tloc_1 + tloc_2)
