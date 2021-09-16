import fronts
import json
import xarray as xr
import numpy as np


def create_data(data):
    d = np.array(data)
    d1 = np.arange(d.shape[0])
    d2 = np.arange(d.shape[1])
    return d, d1, d2


def test_no_zeropoints():
    d, d1, d2 = create_data([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert fronts.zeropoints(d, d1, d2).size == 0


def test_zeropoints_on_grid():
    d, d1, d2 = create_data([[-1, 0, 1], [-1, np.nan, 1], [-1, np.nan, 1]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[0, 1]])
    np.testing.assert_allclose(res, expected)


def test_zeropoints_between_grid():
    d, d1, d2 = create_data([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[0, 0.5], [1, 0.5], [2, 0.5]])
    np.testing.assert_allclose(res, expected)


def test_zeropoints_on_grid_y():
    d, d1, d2 = create_data([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[1, 0], [1, 1], [1, 2]])
    np.testing.assert_allclose(res, expected)


def test_zeropoints_between_grid_y():
    d, d1, d2 = create_data([[-1, -1, -1], [1, 1, 1], [1, 1, 1]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[0.5, 0], [0.5, 1], [0.5, 2]])
    np.testing.assert_allclose(res, expected)


def test_zeropoints_fractional():
    d, d1, d2 = create_data([[-1, -1, -1], [3, 3, 3], [1, 1, 1]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[0.25, 0], [0.25, 1], [0.25, 2]])
    np.testing.assert_allclose(res, expected)


def test_zeropoints_diagonal():
    d, d1, d2 = create_data([[-3, -1, 2], [-2, 0, 1], [1, 1, 4]])
    res = fronts.zeropoints(d, d1, d2)
    expected = np.array([[1, 1], [0, 4 / 3], [5 / 3, 0]])
    np.testing.assert_allclose(res, expected)


def test_era5():

    tafile = xr.open_dataarray(
        "/g/data/rt52/era5/pressure-levels/reanalysis/t/2020/t_era5_oper_pl_20200101-20200131.nc"
    )
    uafile = xr.open_dataarray(
        "/g/data/rt52/era5/pressure-levels/reanalysis/u/2020/u_era5_oper_pl_20200101-20200131.nc"
    )
    vafile = xr.open_dataarray(
        "/g/data/rt52/era5/pressure-levels/reanalysis/v/2020/v_era5_oper_pl_20200101-20200131.nc"
    )
    husfile = xr.open_dataarray(
        "/g/data/rt52/era5/pressure-levels/reanalysis/q/2020/q_era5_oper_pl_20200101-20200131.nc"
    )

    n = 0

    # 32 index in level --> 900 hPa
    ta = tafile[n, 32, 0:100, 0:100]
    ua = uafile[n, 32, 0:100, 0:100]
    va = vafile[n, 32, 0:100, 0:100]
    hus = husfile[n, 32, 0:100, 0:100]

    t_wet = fronts.wetbulb(ta, hus, 900, steps=120)

    frontdata = fronts.front(
        t_wet, ua, va, threshold_i=-1e-10, numsmooth=9, minlength=50
    )

    timestring = np.datetime_as_string(tafile.time.data[n], unit="h")

    with open("tests/test_fronts.json", "w") as outfile:
        json.dump(frontdata, outfile)

    with open(f"tests/900hPa_fronts_{timestring}.json") as sample_file:
        sample = json.load(sample_file)

        assert frontdata == sample
