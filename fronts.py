# module imports
import numpy as np
import scipy.signal
import xarray as xr
import scipy.spatial.distance as sp_dist
import scipy.spatial
import geopy.distance as gp_dist

def wetbulb(ta,hus,plev,steps=100,ta_units=None):
	# calculates wetbulb temperature from pressure-level data
	# Inputs: ta - temperature field (xarray)
	#         hus - specific humidity field (xarray)
	#         plev - the level of the data (in hPa, scalar)
	#         steps -  the number of steps in the numerical calculation
	#         ta_units - the units of the temperature field (if not provided, read from ta)
	if ta_units==None:
		ta_units=ta.units
	#saturation vapor pressure
	if ta_units=='K' or ta_units=='Kelvin' or ta_units=='kelvin':
		es=6.1094*np.exp((17.625*(ta-273.15))/(ta-30.11))
	elif ta_units=='C' or ta_units=='degC' or ta_units=='deg_C' or ta_units=='Celcius' or ta_units=='celcius':
		es=6.1094*np.exp((17.625*(ta))/(ta+243.04))
	else:
		raise ValueError("Input temperature unit not recognised, use Kelvin (K) or Celcius (C, degC, deg_C)")
	#relative humidity from specific humidity and sat. vap. pres.
	rh=((hus*(plev-es))/(0.622*(1-hus)*es))
	# vapor pressure
	e=es*rh
	# dewpoint temperature
	t_dewpoint=((243.5*np.log(e/6.112))/(17.67-np.log(e/6.112)))+273.15
	
	#unlike the above, calculating the wetbulb temperature is done numerically
	delta_t=(ta-t_dewpoint)/steps
	cur_diff=np.abs(es-e)
	t_wet=ta.copy()

	for i in range(steps):
		cur_t = ta-i*delta_t
		es_cur_t=6.1094*np.exp((17.625*(cur_t-273.15))/(cur_t-30.11))
		adiabatic_adj=850*(ta-cur_t)*(.00066)*(1+.00115*cur_t)
		diff=np.abs(es_cur_t-adiabatic_adj-e)
		t_wet.data[diff<cur_diff]=cur_t.data[diff<cur_diff]
		cur_diff.data[diff<cur_diff]=diff.data[diff<cur_diff]
	
	return t_wet

def dewpoint(ta,hus,plev,ta_units=None):
	# calculates depoint temperature from pressure-level data
	# Inputs: ta - temperature field (xarray)
	#         hus - specific humidity field (xarray)
	#         plev - the level of the data (in hPa, scalar)
	#         ta_units - the units of the temperature field (if not provided, read from ta)
	if ta_units==None:
		ta_units=ta.units
	if ta_units=='K' or ta_units=='Kelvin' or ta_units=='kelvin':
		es=6.1094*np.exp((17.625*(ta-273.15))/(ta-30.11))
	elif ta_units=='C' or ta_units=='degC' or ta_units=='deg_C' or ta_units=='Celcius' or ta_units=='celcius':
		es=6.1094*np.exp((17.625*(ta))/(ta+243.04))
	else:
		raise ValueError("Input temperature unit not recognised, use Kelvin (K) or Celcius (C, degC, deg_C)")
	rh=((hus*(plev-es))/(0.622*(1-hus)*es))
	e=es*rh
	t_dewpoint=((243.5*np.log(e/6.112))/(17.67-np.log(e/6.112)))+273.15
	return t_dewpoint

def zeropoints(data,dim1,dim2):
	# finds zero-crossing points in a gridded data set along the lines of each dimension
	# inputs: data - 2d data field (numpy array)
	#         dim1 - coords of the first dim of data (np array)
	#         dim2 - coords of the second dim of data (np array)
	n1,n2 = data.shape
	#assuming regularly spaced grid:
	d_dim2=dim2[1]-dim2[0]
	d_dim1=dim1[1]-dim1[0]
	tloc_1=[]
	tloc_2=[]
	for lonn in range(0,n2-1):
		for latn in range(0,n1-1):
			flag=False
			if data[latn,lonn]==0:
				tloc_1.append([dim1[latn],dim2[lonn]])
				flag=True
			else:
				if np.isfinite(data[latn,lonn]) and np.isfinite(data[latn,lonn+1]) and not flag:
					if (data[latn,lonn]>0 and data[latn,lonn+1]<0) or (data[latn,lonn]<0 and data[latn,lonn+1]>0):
						tloc_1.append([dim1[latn],dim2[lonn]+d_dim2*np.abs(data[latn,lonn]/(data[latn,lonn]-data[latn,lonn+1]))])
				if np.isfinite(data[latn,lonn]) and np.isfinite(data[latn+1,lonn]) and not flag:
					if (data[latn,lonn]>0 and data[latn+1,lonn]<0) or (data[latn,lonn]<0 and data[latn+1,lonn]>0):
						tloc_2.append([dim1[latn]+d_dim1*np.abs(data[latn,lonn]/(data[latn,lonn]-data[latn+1,lonn])),dim2[lonn]])
	return np.array(tloc_1+tloc_2)
	
def frontfields(data,ua,va,threshold=-0.3e-10):
	# returns a field where zero crossings indicate fronts, along with an indicator of front speed
	# and magnitude
	# INPUTS: data - field to find fronts on (2d xarray)
	#         ua - zonal winds on same surface as data (xarray)
	#         va - meridional winds on data surface (xarray)
	#         threshold - intensity threshold for the fronts (needs to be less than zero)
	erad=6371e3
	if 'lon' in data.dims:
		dtdy=data.differentiate('lat')*180/(np.pi*erad)
		dtdx=data.differentiate('lon')*180/(np.pi*erad*xr.ufuncs.cos(data.lat*np.pi/180))
		mag=xr.ufuncs.sqrt(dtdy**2+dtdx**2)
		dmagdy=mag.differentiate('lat')*180/(np.pi*erad)
		dmagdx=mag.differentiate('lon')*180/(np.pi*erad*xr.ufuncs.cos(mag.lat*np.pi/180))
		fr_func=((dtdx*dmagdx)+(dtdy*dmagdy))/mag
		maggradmag=xr.ufuncs.sqrt(dmagdy**2+dmagdx**2)
		fr_speed=(ua*dmagdx+va*dmagdy)/maggradmag
		mgmdy=dmagdy.differentiate('lat')*180/(np.pi*erad)
		mgmdx=dmagdx.differentiate('lon')*180/(np.pi*erad*xr.ufuncs.cos(mag.lat*np.pi/180))
	else:
		dtdy=data.differentiate('latitude')*180/(np.pi*erad)
		dtdx=data.differentiate('longitude')*180/(np.pi*erad*xr.ufuncs.cos(data.latitude*np.pi/180))
		mag=xr.ufuncs.sqrt(dtdy**2+dtdx**2)
		dmagdy=mag.differentiate('latitude')*180/(np.pi*erad)
		dmagdx=mag.differentiate('longitude')*180/(np.pi*erad*xr.ufuncs.cos(mag.latitude*np.pi/180))
		fr_func=((dtdx*dmagdx)+(dtdy*dmagdy))/mag
		maggradmag=xr.ufuncs.sqrt(dmagdy**2+dmagdx**2)
		fr_speed=(ua*dmagdx+va*dmagdy)/maggradmag
		mgmdy=dmagdy.differentiate('latitude')*180/(np.pi*erad)
		mgmdx=dmagdx.differentiate('longitude')*180/(np.pi*erad*xr.ufuncs.cos(mag.latitude*np.pi/180))
	loc=mgmdy+mgmdx
	loc.data[fr_func>0]=np.nan
	loc.data[fr_func>threshold]=np.nan
	return loc,fr_speed,mag
	
def linejoin(inpts,searchdist=1.5,minlength=250,lonex=0):
	# turns a list of lat-lon points into a list of joined lines
	# INPUTS: inpts - the list of points (list of lat-lon points)
	#         searchdist - degree radius around each point that other points within are
	#                      deemed to be part of the same line
	#         minlength - minimum end-to-end length of the lines (in km)
	#         lonex - minimum end-to-end longitudinal extent
	ptcount=inpts.shape[0]
	not_used=np.ones((ptcount),dtype=bool)

	lines=[]
	nrec=[]
	na=0
	for ii in range(ptcount):
		if not_used[ii]:
			print(ii,"/",ptcount)
			templat=[]
			templon=[]
			templat2=[]
			templon2=[]
			templat.append(inpts[ii,0])
			templon.append(inpts[ii,1])
			not_used[ii]=False
			t=ii
			insearchdist=True
			while insearchdist:
				mindist=np.inf
				for jj in range(ptcount):
					if not_used[jj]:
						dist=sp_dist.euclidean((inpts[t]),(inpts[jj]))
						if dist>0 and dist<mindist:
							mindist=dist
							rec=jj
							distr=dist
				#have found nearest unused point
				if mindist<searchdist:
					not_used[rec]=False
					templat.append(inpts[rec,0])
					templon.append(inpts[rec,1])
					t=rec
				else:
					insearchdist=False
			# search other direction
			t=ii
			insearchdist=True
			while insearchdist:
				mindist=np.inf
				for jj in range(ptcount):
					if not_used[jj]:
						dist=sp_dist.euclidean((inpts[t]),(inpts[jj]))
						if dist>0 and dist<mindist:
							mindist=dist
							rec=jj
							distr=dist
				#have found nearest unused point
				if mindist<searchdist:
					not_used[rec]=False
					templat2.append(inpts[rec,0])
					templon2.append(inpts[rec,1])
					t=rec
				else:
					insearchdist=False
			if len(templat2)>0:
				templat=templat2[::-1]+templat
				templon=templon2[::-1]+templon
			lines.append((templat,templon))
			nrec.append(len(templat))
	print("lines found:",len(lines))
	filt_lines=[]
	for line in lines:
		ln_dist=gp_dist.distance((line[0][0],line[1][0]),(line[0][-1],line[1][-1])).km
		lon_extent=max(line[1])-min(line[1])
		if ln_dist>minlength and lon_extent>lonex:
			filt_lines.append(line)
	lines=filt_lines
	return lines


def linejoin_graph(inpts, searchdist: float=1.5, minlength: float=250, lonex: float=0):
    """
    Turns a list of lat-lon points into a list of joined lines

    Args:
        inpts - the list of points (list of lat-lon points)
        searchdist - degree radius around each point that other points within are
                     deemed to be part of the same line
        minlength - minimum end-to-end length of the lines (in km)
        lonex - minimum end-to-end longitudinal extent

    Returns:
        List with each member a tuple of (lats, lons) for each line found
    """

    tree = scipy.spatial.KDTree(inpts)

    assert tree.m == 2, f"Expected 2d input points, found {tree.m}d"

    distances = tree.sparse_distance_matrix(tree, searchdist)

    # Cull to only the minimal connections
    span = scipy.sparse.csgraph.minimum_spanning_tree(distances)

    # Get the components
    ncomp, comp_labels = scipy.sparse.csgraph.connected_components(span)

    lines = []

    for c in range(ncomp):
        comp_indices = np.arange(comp_labels.size)[comp_labels == c]
        if len(comp_indices) <= 1:
            continue

        start = comp_indices[1]

        # This ordering may not start at the start of the span, but it will end at the end of it
        order = scipy.sparse.csgraph.depth_first_order(span, start, return_predecessors=False, directed=False)

        # Search the other way to get the correct ordering for the full path
        order = scipy.sparse.csgraph.depth_first_order(span, order[-1], return_predecessors=False, directed=False)

        # Add the coordinates from this path to the output list
        lines.append((inpts[order,0], inpts[order,1]))

    # Filter from the previous version
    filt_lines=[]
    for line in lines:
        ln_dist=gp_dist.distance((line[0][0],line[1][0]),(line[0][-1],line[1][-1])).km
        lon_extent=max(line[1])-min(line[1])
        if ln_dist>minlength and lon_extent>lonex:
            filt_lines.append(line)
    lines=filt_lines

    return lines

	
def smoother(data,numsmooth=9,smooth_kernel=np.ones((3,3))/9):
	# smooths an input 2-d xarray using a given kernel and number of passes
	smoothfunc=lambda x:scipy.signal.convolve2d(x,smooth_kernel,mode='same',boundary='symm')
	smoothcount=0
	while smoothcount<numsmooth:
		if smoothcount==0:
			output=xr.apply_ufunc(smoothfunc,data)
		else:
			output=xr.apply_ufunc(smoothfunc,output)
		smoothcount+=1
	
	return output

def front(data,u,v,threshold_i=-0.3e-10,threshhold_s=1.5,numsmooth=3,smooth_kernel=np.ones((3,3))/9,minlength=250):
	#identifies fronts in data using Berry et al. method
	# INPUTS: data         - field to find fronts upon
	#         u            - zonal wind on the same surface as data
	#         v            - meridional wind on the same surface as data
    #         threshold_i  - intensity threshold for fronts
	#         threshold_s  - speed threshold for cold/warm fronts
	#         numsmooth    - number of passes of the smoothing kernel
	#         smooth_kernel - the smoothing kernel
	#         minlength     - minimum end-to-end front length (km)
	# define internal constants/parameters here
	re=6371e3

	# smooth input data
	smoothfunc=lambda x:scipy.signal.convolve2d(x,smooth_kernel,mode='same',boundary='symm')
	smoothcount= 0
	while smoothcount<numsmooth:
		data = xr.apply_ufunc(smoothfunc,data)
		u = xr.apply_ufunc(smoothfunc,u)
		v = xr.apply_ufunc(smoothfunc,v)
		smoothcount+=1
	
	loc,fr_speed,mag=frontfields(data,u,v,threshold_i)
	if 'lon' in data.dims:
		out=zeropoints(loc.data,loc.lat.data,loc.lon.data)
	else:
		out=zeropoints(loc.data,loc.latitude.data,loc.longitude.data)
	
	lats=xr.DataArray(out[:,0],dims='pts')
	lons=xr.DataArray(out[:,1],dims='pts')
	if 'lon' in data.dims:
		spdloc=fr_speed.interp(lat=lats,lon=lons).data
	else:
		spdloc=fr_speed.interp(latitude=lats,longitude=lons).data
	n_pts=out.shape[0]
	cpts=[]
	wpts=[]
	spts=[]
	
	for n in range(n_pts):
		if spdloc[n]<-1*threshhold_s:
			cpts.append(out[n])
		elif spdloc[n]>threshhold_s:
			wpts.append(out[n])
		elif np.isfinite(spdloc[n]):
			spts.append(out[n])
	cpts=np.array(cpts)
	wpts=np.array(wpts)
	spts=np.array(spts)
	
	clines=linejoin(cpts,minlength=minlength)
	wlines=linejoin(wpts,minlength=minlength)
	slines=linejoin(spts,minlength=minlength)
	
	clinemag=[]
	wlinemag=[]
	slinemag=[]
	print('adding gradient')
	for line in clines:
		x=list(line)
		if 'lon' in data.dims:
			magline=mag.interp(lat=xr.DataArray(x[0],dims='pts'),lon=xr.DataArray(x[1],dims='pts')).data
		else:
			magline=mag.interp(latitude=xr.DataArray(x[0],dims='pts'),longitude=xr.DataArray(x[1],dims='pts')).data
		x.append(magline.tolist())
		clinemag.append(x)
	for line in wlines:
		x=list(line)
		if 'lon' in data.dims:
			magline=mag.interp(lat=xr.DataArray(x[0],dims='pts'),lon=xr.DataArray(x[1],dims='pts')).data
		else:
			magline=mag.interp(latitude=xr.DataArray(x[0],dims='pts'),longitude=xr.DataArray(x[1],dims='pts')).data
		x.append(magline.tolist())
		wlinemag.append(x)
	for line in slines:
		x=list(line)
		if 'lon' in data.dims:
			magline=mag.interp(lat=xr.DataArray(x[0],dims='pts'),lon=xr.DataArray(x[1],dims='pts')).data
		else:
			magline=mag.interp(latitude=xr.DataArray(x[0],dims='pts'),longitude=xr.DataArray(x[1],dims='pts')).data
		x.append(magline.tolist())
		slinemag.append(x)
	
	# not certain if there's a better output format for the data
	frontdata={'cold_fronts':clinemag,'warm_fronts':wlinemag,'stationary_fronts':slinemag}
	
	return frontdata
