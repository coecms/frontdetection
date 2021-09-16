# module imports
import numpy as np
from numpy.lib.utils import _lookfor_generate_cache
import scipy.signal
import scipy.interpolate as interp
import xarray as xr
import scipy.spatial.distance as sp_dist
import geopy.distance as gp_dist


def calculate_saturation_pressure(ta):
    """
    (xarray.DataArray) -> xarray.DataArray

    Calculates the saturation pressure.

    Argument:
    ta:       Temperature Array in Kelvin
    """
    return 6.1094 * np.exp((17.625 * (ta - 273.15)) / (ta - 30.11))


def calculate_relative_humidity(hus, plev, es):
    """
    Calculate the relative humidity

    Arguments:
    hus:        specific humidity
    plev:       level in hPa
    es:         saturation pressure
    """
    return (hus * (plev - es)) / (0.622 * (1 - hus) * es)


def calculate_dewpoint_temperature(e):
    """
    Calculates the dewpoint from the vapour pressure

    Input:
    e:      Vapour Pressure
    """
    return ((243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))) + 273.15


def calculate_atmospheric_fields(ta, hus, plev, ta_units=None):
    """
    (DataArray, DataArray, Numeric, Str) -> DataArray, DataArray, DataArray, DataArray
    
    Calculates the Saturation Pressure, the Vapour Pressure, 
    the Relative Humidity, and the Dewpoint Temperature.

    Inputs:
    ta:         Temperature
    hus:        Specific Humidity
    plev:       Level of Data (hPa)
    ta_units:   (Optional) units of ta. Defaults to ta.units.

    Output:
    Tuple(Saturation Pressure, Vapour Pressure, Relative Humidity, Dewpoint Temperature)
    """
    if ta_units is None:
        ta_units = ta.units
    # saturation vapor pressure
    if ta_units.lower() in ["k", "kelvin"]:
        es = calculate_saturation_pressure(ta)
    elif ta_units.lower() in ["c", "degc", "deg_c", "celsius"]:
        es = calculate_saturation_pressure(ta+273.15)
    else:
        raise ValueError(
            "Input temperature unit not recognised, use Kelvin (K) or Celcius (C, degC, deg_C)"
        )
    # relative humidity from specific humidity and sat. vap. pres.
    rh = calculate_relative_humidity(hus, plev, es)
    # vapor pressure
    e = es * rh
    # dewpoint temperature
    t_dewpoint = calculate_dewpoint_temperature(e)

    return (es, rh, e, t_dewpoint)


def wetbulb(ta, hus, plev, steps=100, ta_units=None):
    """
    calculates wetbulb temperature from pressure-level data
    Inputs: ta - temperature field (xarray)
            hus - specific humidity field (xarray)
            plev - the level of the data (in hPa, scalar)
            steps -  the number of steps in the numerical calculation
            ta_units - the units of the temperature field (if not provided, read from ta)
    """

    es, _, e, t_dewpoint = calculate_atmospheric_fields(ta, hus, plev, ta_units)

    # unlike the above, calculating the wetbulb temperature is done numerically
    delta_t = (ta - t_dewpoint) / steps
    cur_diff = np.abs(es - e)
    t_wet = ta.copy()

    for i in range(steps):
        cur_t = ta - i * delta_t
        es_cur_t = calculate_saturation_pressure(cur_t)
        adiabatic_adj = 850 * (ta - cur_t) * (0.00066) * (1 + 0.00115 * cur_t)
        diff = np.abs(es_cur_t - adiabatic_adj - e)
        t_wet.data[diff < cur_diff] = cur_t.data[diff < cur_diff]
        cur_diff.data[diff < cur_diff] = diff.data[diff < cur_diff]

    return t_wet


def dewpoint(ta, hus, plev, ta_units=None):
    """
    calculates depoint temperature from pressure-level data
    Inputs: ta - temperature field (xarray)
            hus - specific humidity field (xarray)
            plev - the level of the data (in hPa, scalar)
            ta_units - the units of the temperature field (if not provided, read from ta)
    """

    _, _, _, t_dewpoint = calculate_atmospheric_fields(ta, hus, plev, ta_units)

    return t_dewpoint


def zeropoints(data, dim1, dim2):
    """
    finds zero-crossing points in a gridded data set along the lines of each dimension
    inputs: data - 2d data field (numpy array)
            dim1 - coords of the first dim of data (np array)
            dim2 - coords of the second dim of data (np array)
    """

    ## Find points where the value itself is zero:
    zero_locations = [
        [dim1[idx1], dim2[idx2]] 
        for idx1, idx2 in zip(*np.where(data==0))
    ]

    ## Find zeropoints along latitude
    for dim1_val, dim2_data in zip(dim1, data):
        # Multiply each data point with the next. Negative values then indicate change in sign
        indicator_array = dim2_data[:-1] * dim2_data[1:]
        zero_locations.extend([
            [dim1_val, interp.interp1d(dim2_data[i:i+2], dim2[i:i+2])(0)] 
            for i in np.where(indicator_array < 0)[0]
        ])

    for dim2_val, dim1_data in zip(dim2, data.T):
        # Multiply each data point with the next. Negative values then indicate change in sign
        indicator_array = dim1_data[:-1] * dim1_data[1:]
        zero_locations.extend([
            [interp.interp1d(dim1_data[i:i+2], dim1[i:i+2])(0), dim2_val]
            for i in np.where(indicator_array < 0)[0]
        ])

    return np.array(zero_locations)


def frontfields(data, ua, va, threshold=-0.3e-10):
    # returns a field where zero crossings indicate fronts, along with an indicator of front speed
    # and magnitude
    # INPUTS: data - field to find fronts on (2d xarray)
    #         ua - zonal winds on same surface as data (xarray)
    #         va - meridional winds on data surface (xarray)
    #         threshold - intensity threshold for the fronts (needs to be less than zero)
    erad = 6371e3
    if "lon" in data.dims:
        dtdy = data.differentiate("lat") * 180 / (np.pi * erad)
        dtdx = (
            data.differentiate("lon")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(data.lat * np.pi / 180))
        )
        mag = xr.ufuncs.sqrt(dtdy ** 2 + dtdx ** 2)
        dmagdy = mag.differentiate("lat") * 180 / (np.pi * erad)
        dmagdx = (
            mag.differentiate("lon")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(mag.lat * np.pi / 180))
        )
        fr_func = ((dtdx * dmagdx) + (dtdy * dmagdy)) / mag
        maggradmag = xr.ufuncs.sqrt(dmagdy ** 2 + dmagdx ** 2)
        fr_speed = (ua * dmagdx + va * dmagdy) / maggradmag
        mgmdy = dmagdy.differentiate("lat") * 180 / (np.pi * erad)
        mgmdx = (
            dmagdx.differentiate("lon")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(mag.lat * np.pi / 180))
        )
    else:
        dtdy = data.differentiate("latitude") * 180 / (np.pi * erad)
        dtdx = (
            data.differentiate("longitude")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(data.latitude * np.pi / 180))
        )
        mag = xr.ufuncs.sqrt(dtdy ** 2 + dtdx ** 2)
        dmagdy = mag.differentiate("latitude") * 180 / (np.pi * erad)
        dmagdx = (
            mag.differentiate("longitude")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(mag.latitude * np.pi / 180))
        )
        fr_func = ((dtdx * dmagdx) + (dtdy * dmagdy)) / mag
        maggradmag = xr.ufuncs.sqrt(dmagdy ** 2 + dmagdx ** 2)
        fr_speed = (ua * dmagdx + va * dmagdy) / maggradmag
        mgmdy = dmagdy.differentiate("latitude") * 180 / (np.pi * erad)
        mgmdx = (
            dmagdx.differentiate("longitude")
            * 180
            / (np.pi * erad * xr.ufuncs.cos(mag.latitude * np.pi / 180))
        )
    loc = mgmdy + mgmdx
    loc.data[fr_func > 0] = np.nan
    loc.data[fr_func > threshold] = np.nan
    return loc, fr_speed, mag


def follow_line(inpts, initial_index, searchdist, used, templat=[], templon=[]):
    """
    Builds a line of nearest neighbours, starting with inpts[initial_index], and returns
    their longitudes and latitudes in a tuple of two lists.
    """
    ptcount = len(used)
    current_index = initial_index
    while True:
        shortest_distance_so_far = np.inf
        for comparison_index in range(ptcount):
            ## Only look at points that aren't yet
            ## part of any line.
            if used[comparison_index]:
                continue

            distance = sp_dist.euclidean((inpts[current_index]), (inpts[comparison_index]))

            ## Check if distance to new point is shorter
            ## than previous nearest
            if 0 < distance < shortest_distance_so_far:
                shortest_distance_so_far = distance
                nearest_neighbor_index = comparison_index

        ## If nearest point is too far away, we've reached the end
        ## of this side of the line.
        if shortest_distance_so_far >= searchdist:
            break

        ## Since it wasn't too far away, add it to the current line,
        ## and then check again from here.
        used[nearest_neighbor_index] = True
        templat.append(inpts[nearest_neighbor_index, 0])
        templon.append(inpts[nearest_neighbor_index, 1])
        current_index = nearest_neighbor_index   
    return templat, templon


def line_filter(line, min_length, min_lon_extend):
    """(List of List of float, float, float) -> bool
    Checks whether line adheres to two rules:
    1. The total distance in km between the first and last
       point on the line must be at least min_length.
    2. The minimum extension in longitude must exceed min_lon_extend
    """
    lon_extent = max(line[1]) - min(line[1])
    if lon_extent < min_lon_extend:
        return False

    ln_dist = gp_dist.distance(
        (line[0][0], line[1][0]), (line[0][-1], line[1][-1])
    ).km
    return ln_dist > min_length


def linejoin(inpts, searchdist=1.5, minlength=250, lonex=0):
    # turns a list of lat-lon points into a list of joined lines
    # INPUTS: inpts - the list of points (list of lat-lon points)
    #         searchdist - degree radius around each point that other points within are
    #                      deemed to be part of the same line
    #         minlength - minimum end-to-end length of the lines (in km)
    #         lonex - minimum end-to-end longitudinal extent
    ptcount = inpts.shape[0]
    used = np.zeros((ptcount), dtype=bool)

    lines = []

    for initial_index in range(ptcount):

        ## If the current point already is part of a line, ignore it.
        if used[initial_index]:
            continue

        print(initial_index, "/", ptcount)

        used[initial_index] = False

        templat, templon = follow_line(
            inpts, initial_index, searchdist, used, 
            [inpts[initial_index, 0]], [inpts[initial_index, 1]])

        # search other direction
        templat2, templon2 = follow_line(inpts, initial_index, searchdist, used, [], [])

        ## If there actually was a second direction, add them
        ## in reverse order.
        if len(templat2) > 0:
            templat = templat2[::-1] + templat
            templon = templon2[::-1] + templon

        lines.append((templat, templon))

    print("lines found:", len(lines))
    lines = [l for l in lines if line_filter(l, minlength, lonex)]
    print(f"lines remaining after filter: {len(lines)}")
    return lines


def smoother(data, numsmooth=9, smooth_kernel=np.ones((3, 3)) / 9):
    # smooths an input 2-d xarray using a given kernel and number of passes
    smoothfunc = lambda x: scipy.signal.convolve2d(
        x, smooth_kernel, mode="same", boundary="symm"
    )
    smoothcount = 0
    while smoothcount < numsmooth:
        if smoothcount == 0:
            output = xr.apply_ufunc(smoothfunc, data)
        else:
            output = xr.apply_ufunc(smoothfunc, output)
        smoothcount += 1

    return output


def front(
    data,
    u,
    v,
    threshold_i=-0.3e-10,
    threshhold_s=1.5,
    numsmooth=3,
    smooth_kernel=np.ones((3, 3)) / 9,
    minlength=250,
):
    # identifies fronts in data using Berry et al. method
    # INPUTS: data         - field to find fronts upon
    #         u            - zonal wind on the same surface as data
    #         v            - meridional wind on the same surface as data
    #         threshold_i  - intensity threshold for fronts
    #         threshold_s  - speed threshold for cold/warm fronts
    #         numsmooth    - number of passes of the smoothing kernel
    #         smooth_kernel - the smoothing kernel
    #         minlength     - minimum end-to-end front length (km)
    # define internal constants/parameters here
    re = 6371e3

    # smooth input data
    smoothfunc = lambda x: scipy.signal.convolve2d(
        x, smooth_kernel, mode="same", boundary="symm"
    )
    smoothcount = 0
    while smoothcount < numsmooth:
        data = xr.apply_ufunc(smoothfunc, data)
        u = xr.apply_ufunc(smoothfunc, u)
        v = xr.apply_ufunc(smoothfunc, v)
        smoothcount += 1

    loc, fr_speed, mag = frontfields(data, u, v, threshold_i)
    if "lon" in data.dims:
        out = zeropoints(loc.data, loc.lat.data, loc.lon.data)
    else:
        out = zeropoints(loc.data, loc.latitude.data, loc.longitude.data)

    lats = xr.DataArray(out[:, 0], dims="pts")
    lons = xr.DataArray(out[:, 1], dims="pts")
    if "lon" in data.dims:
        spdloc = fr_speed.interp(lat=lats, lon=lons).data
    else:
        spdloc = fr_speed.interp(latitude=lats, longitude=lons).data
    n_pts = out.shape[0]
    cpts = []
    wpts = []
    spts = []

    for n in range(n_pts):
        if spdloc[n] < -1 * threshhold_s:
            cpts.append(out[n])
        elif spdloc[n] > threshhold_s:
            wpts.append(out[n])
        elif np.isfinite(spdloc[n]):
            spts.append(out[n])
    cpts = np.array(cpts)
    wpts = np.array(wpts)
    spts = np.array(spts)

    clines = linejoin(cpts, minlength=minlength)
    wlines = linejoin(wpts, minlength=minlength)
    slines = linejoin(spts, minlength=minlength)

    clinemag = []
    wlinemag = []
    slinemag = []
    print("adding gradient")
    for line in clines:
        x = list(line)
        if "lon" in data.dims:
            magline = mag.interp(
                lat=xr.DataArray(x[0], dims="pts"), lon=xr.DataArray(x[1], dims="pts")
            ).data
        else:
            magline = mag.interp(
                latitude=xr.DataArray(x[0], dims="pts"),
                longitude=xr.DataArray(x[1], dims="pts"),
            ).data
        x.append(magline.tolist())
        clinemag.append(x)
    for line in wlines:
        x = list(line)
        if "lon" in data.dims:
            magline = mag.interp(
                lat=xr.DataArray(x[0], dims="pts"), lon=xr.DataArray(x[1], dims="pts")
            ).data
        else:
            magline = mag.interp(
                latitude=xr.DataArray(x[0], dims="pts"),
                longitude=xr.DataArray(x[1], dims="pts"),
            ).data
        x.append(magline.tolist())
        wlinemag.append(x)
    for line in slines:
        x = list(line)
        if "lon" in data.dims:
            magline = mag.interp(
                lat=xr.DataArray(x[0], dims="pts"), lon=xr.DataArray(x[1], dims="pts")
            ).data
        else:
            magline = mag.interp(
                latitude=xr.DataArray(x[0], dims="pts"),
                longitude=xr.DataArray(x[1], dims="pts"),
            ).data
        x.append(magline.tolist())
        slinemag.append(x)

    # not certain if there's a better output format for the data
    frontdata = {
        "cold_fronts": clinemag,
        "warm_fronts": wlinemag,
        "stationary_fronts": slinemag,
    }

    return frontdata
