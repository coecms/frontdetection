import fronts
import json
import xarray as xr
import numpy as np

tafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/t/2020/t_era5_oper_pl_20200101-20200131.nc')
uafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/u/2020/u_era5_oper_pl_20200101-20200131.nc')
vafile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/v/2020/v_era5_oper_pl_20200101-20200131.nc')
husfile=xr.open_dataarray('/g/data/rt52/era5/pressure-levels/reanalysis/q/2020/q_era5_oper_pl_20200101-20200131.nc')

for n in range(tafile.shape[0]):
	# 32 index in level --> 900 hPa
	ta=tafile[n,32]
	ua=uafile[n,32]
	va=vafile[n,32]
	hus=husfile[n,32]
	
	t_wet=fronts.wetbulb(ta,hus,900,steps=120)
	
	frontdata=fronts.front(t_wet,ua,va,threshold_i=-1e-10,numsmooth=9,minlength=50)
	
	timestring=np.datetime_as_string(tafile.time.data[n],unit='h')
	outfile=open('/g/data/w97/mjk563/900hPa_fronts_'+timestring+'.json','w')
	json.dump(frontdata,outfile)
	outfile.close()