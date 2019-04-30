import numpy as np
from linetools.spectra.xspectrum1d import XSpectrum1D


def weight_mean_coadd(cubes, vars, stackwcs, varwcs, outfile=None,
					  outvarfile=None, air_to_vac=True):
	# Go through each cube and find the brightest pixels
	buff = 200
	levels = np.zeros(len(cubes))

	for i,cc in enumerate(cubes):
		var= vars[i]
		partdat = cc[buff:-buff, :, :]
		sumdat = np.sum(partdat, axis=0)
		partvar = var[buff:-buff, :, :]
		sumvar =  np.sum(partvar,axis=0)
		datnonan = sumdat[(~np.isnan(sumdat)&(sumdat!=0))]
		varnonan = sumvar[(~np.isnan(sumdat)&(sumdat!=0))]
		mednonan = np.median(datnonan)
		stddev = np.std(datnonan)
		#bright = np.where((datnonan > (mednonan+stddev))&(datnonan>0))[0]
		bright = np.where((datnonan > (mednonan)) & (datnonan > 0))[0]
		sumbright = np.sum(datnonan[bright])
		varbright = np.sum(varnonan[bright])
		cc[np.isnan(cc)] = 0.
		var[np.isnan(var)] = 0.

		sn = sumbright / np.sqrt(varbright)
		#import pdb; pdb.set_trace()
		print(sumbright,mednonan,stddev,sn)

		if i==0:
			combined = cc*sn
			combvar = var*sn**2
		else:
			combined += cc*sn
			combvar += var*sn**2
		levels[i]=sn

	finaldat = combined/np.sum(levels)
	finalvar = combvar / np.sum(levels)**2
	
	newhdulist = stackwcs.to_fits()
	newhdulist[0].data = finaldat
	if air_to_vac is True:
		dims = np.shape(cubes[0])
		lastwaveidx = np.max(dims)-1
		wave1 = stackwcs.wcs_pix2world([[0,0,0]],1)[0][2]
		wave2 = stackwcs.wcs_pix2world([[0,0,lastwaveidx]],1)[0][2]
		newwavearr = np.arange(wave1,wave2+1e-10,1e-10)
		newspec = XSpectrum1D(newwavearr*1e10,np.zeros_like(newwavearr))
		newspec.meta['airvac']='air'
		newspec.airtovac()
		diffs = newspec.wavelength[1:]-newspec.wavelength[0:-1]
		newhdulist[0].header['CRVAL3']=newspec.wvmin.value
		newhdulist[0].header['CRPIX3'] = 1
		newhdulist[0].header['CD3_3']=np.median(diffs.value)
		newhdulist[0].header['PC3_3']=np.median(diffs.value)

		newhdulist[0].header.set('CTYPE3','VAC','Vacuum wavelengths')
		newhdulist[0].header['CUNIT3']='Angstrom'
	if outfile is not None:
		newhdulist.writeto(outfile,overwrite=True)

	varhdulist = varwcs.to_fits()
	varhdulist[0].data = finalvar
	if air_to_vac is True:
		dims = np.shape(cubes[0])
		lastwaveidx = np.max(dims) - 1
		wave1 = varwcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
		wave2 = varwcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
		newwavearr = np.arange(wave1, wave2 + 1e-10, 1e-10)
		newspec = XSpectrum1D(newwavearr * 1e10, np.zeros_like(newwavearr))
		newspec.meta['airvac'] = 'air'
		newspec.airtovac()
		diffs = newspec.wavelength[1:] - newspec.wavelength[0:-1]
		varhdulist[0].header['CRVAL3'] = newspec.wvmin.value
		newhdulist[0].header['CRPIX3'] = 1
		varhdulist[0].header['CD3_3'] = np.median(diffs.value)
		varhdulist[0].header['PC3_3'] = np.median(diffs.value)
		varhdulist[0].header.set('CTYPE3', 'VAC', 'Vacuum wavelengths')
		varhdulist[0].header['CUNIT3'] = 'Angstrom'
	if outvarfile is not None:
		varhdulist.writeto(outvarfile, overwrite=True)
	return newhdulist


def median_coadd(cubes, stackwcs, outfile=None, air_to_vac=True):
	nc_arr = np.array(cubes)
	nc_med = np.median(nc_arr,axis=0)
	newhdulist = stackwcs.to_fits()
	newhdulist[0].data = nc_med
	if air_to_vac is True:
		dims = np.shape(cubes[0])
		lastwaveidx = np.max(dims)-1
		wave1 = stackwcs.wcs_pix2world([[0,0,0]],1)[0][2]
		wave2 = stackwcs.wcs_pix2world([[0,0,lastwaveidx]],1)[0][2]
		newwavearr = np.arange(wave1,wave2+1e-10,1e-10)
		newspec = XSpectrum1D(newwavearr*1e10,np.zeros_like(newwavearr))
		newspec.meta['airvac']='air'
		newspec.airtovac()
		diffs = newspec.wavelength[1:]-newspec.wavelength[0:-1]
		newhdulist[0].header['CRVAL3']=newspec.wvmin.value
		newhdulist[0].header['CRPIX3'] = 1
		newhdulist[0].header['CD3_3']=np.median(diffs.value)
		newhdulist[0].header['PC3_3']=np.median(diffs.value)
		newhdulist[0].header.set('CTYPE3','VAC','Vacuum wavelengths')
		newhdulist[0].header['CUNIT3']='Angstrom'
	if outfile is not None:
		newhdulist.writeto(outfile,overwrite=True)
	return newhdulist

def variance_coadd(varcubes, stackwcs, outfile=None, air_to_vac=True):
	vc_arr = np.array(varcubes)
	### TODO: Need to change this to use a bootstrap error or something!
	vc_med = np.median(vc_arr,axis=0)
	newhdulist = stackwcs.to_fits()
	newhdulist[0].data = vc_med
	if air_to_vac is True:
		dims = np.shape(varcubes[0])
		lastwaveidx = np.max(dims)-1
		wave1 = stackwcs.wcs_pix2world([[0,0,0]],1)[0][2]
		wave2 = stackwcs.wcs_pix2world([[0,0,lastwaveidx]],1)[0][2]
		newwavearr = np.arange(wave1,wave2+1e-10,1e-10)
		newspec = XSpectrum1D(newwavearr*1e10,np.zeros_like(newwavearr))
		newspec.meta['airvac']='air'
		newspec.airtovac()
		diffs = newspec.wavelength[1:]-newspec.wavelength[0:-1]
		newhdulist[0].header['CRVAL3']=newspec.wvmin.value
		newhdulist[0].header['CD3_3']=np.median(diffs.value)
		newhdulist[0].header['PC3_3']=np.median(diffs.value)
		newhdulist[0].header.set('CTYPE3','VAC','Vacuum wavelengths')
		newhdulist[0].header['CUNIT3']='Angstrom'
	if outfile is not None:
		newhdulist.writeto(outfile,overwrite=True)
	return newhdulist