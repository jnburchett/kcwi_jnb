import contVPmeasure as cvp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from kcwi_jnb import transform as kt
from kcwi_jnb import utils as ku
from kcwi_jnb.cube import DataCube
from goodies import veltrans,closest
from astropy.cosmology import Planck15 as cosmo

def continuum_subtract(cubefile, varcube, wave1, wave2, outfile=None,
                       normalize=False):
    """Fit continuum over a wavelength range in each spaxel and subtract

    Parameters
    ----------
    cubefile
    varcube
    wave1
    wave2
    outfile

    Returns
    -------

    """
    slicedhdu = kt.slice_cube(cubefile, wave1, wave2)
    slicedvar = kt.slice_cube(varcube, wave1, wave2)
    hdr = slicedhdu[0].header
    wcs = WCS(hdr)
    dat = slicedhdu[0].data
    var = slicedvar[0].data
    datshape = np.shape(dat)
    wdelt = hdr['PC3_3']
    dim = np.shape(dat)
    lastwaveidx = dim[0] - 1
    w1 = wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
    w2 = wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
    wavearr = np.arange(w1, w2 + wdelt, wdelt)
    idxgrid = np.meshgrid(range(datshape[1]), range(datshape[2]))
    idxgrid = np.array(idxgrid)
    idxs2iter = np.reshape(idxgrid.T, (datshape[1] * datshape[2], 1, 2))
    for i, idi in enumerate(idxs2iter):
        idx0 = idi[0][0]
        idx1 = idi[0][1]
        thisspec = dat[:, idx0, idx1]
        thisvar = np.sqrt(var[:, idx0, idx1])
        if np.all(np.isnan(thisspec)):
            dat[:, idx0, idx1] = 0
        else:
            try:
                contcoeff, contcovmtx = cvp.initcont(wavearr, thisspec, thisvar, wave1, wave2)
                cont = cvp.evalcont(wavearr, contcoeff)
                if normalize:
                    dat[:, idx0, idx1] = thisspec / cont
                else:
                    dat[:, idx0, idx1] = thisspec - cont
            except:
                dat[:, idx0, idx1] = 0

    slicedhdu[0].data = dat
    if outfile is not None:
        slicedhdu.writeto(outfile, overwrite=True)
    return slicedhdu

def equiv_width(cubefile, varcube, contwave1, contwave2,
                linewave1, linewave2, outfile=None):
    """Calculate the equivalent width over some wavelength range
    """
    normcube = continuum_subtract(cubefile,varcube,contwave1,contwave2)





def median_continuum_subtract(cubefile, contwave1, contwave2,
                              linewave1, linewave2, outfile=None):
    """Find continuum by median flux value and subtract from line regiom
    """
    from astropy.io import fits
    medhdu = kt.narrowband(cubefile, contwave1, contwave2, mode='median')
    linehdu = kt.slice_cube(cubefile, linewave1, linewave2)

    dat = linehdu.data - medhdu.data
    sumdat = np.sum(dat,axis=0)

    medhdu.data = sumdat
    if outfile is not None:
        try:
            fits.writeto(data=medhdu.data, header=medhdu.header,
                         filename=outfile, overwrite=True)
            #medhdu.writeto(outfile, overwrite=True)
        except:
            import pdb; pdb.set_trace()
    return medhdu


def extract_spectrum(cube,pixels,wvslice=None):

    from linetools.spectra.xspectrum1d import XSpectrum1D

    if isinstance(cube,str):
        cube = DataCube(cube)

    if wvslice is not None:
        cube = kt.slice_cube(cube,wvslice[0],wvslice[1])


    dat = cube.data

    # Get rid of NaNs
    dc = dat.copy()
    nans = np.where(np.isnan(dc))
    dc[nans]=0

    # Perform the extraction
    spaxels = []
    for i,px in enumerate(pixels):
        thisspec = dc[:,px[0],px[1]]
        spaxels.append(thisspec)
    spaxels=np.array(spaxels)
    medspec = np.median(spaxels, axis=0)

    # Get the wavelength array
    newwavearr = cube.wavelength

    # Create the spectrum
    try:
        spec = XSpectrum1D(wave = newwavearr,flux=medspec)
    except:
        import pdb; pdb.set_trace()
    return spec

def radial_profile(data, center=None):
    if center is None:
        center = np.where(data==np.max(data))
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def position_velocity(cube,center,regionwidth=3.,regionextent=20.*u.kpc,
                      velocitylimit=1000.,redshift = 0.6942,restwave=2796.3543):
    velarr = veltrans(redshift,cube.wavelength,restwave)
    cp = cube.wcs.wcs_world2pix([[center.ra.deg, center.dec.deg, 1]], 1)  # (ra,dec,lambda)
    ### Get vertical pixel scale
    cencoordsp1 = cube.wcs.wcs_pix2world([[cp[0][0], cp[0][1] + 1, cp[0][2]]], 1)
    ccp1obj = SkyCoord(cencoordsp1[0][0], cencoordsp1[0][1], unit='deg')
    print(center.separation(ccp1obj))
    cencoordsp1 = cube.wcs.wcs_pix2world([[cp[0][0]+1, cp[0][1], cp[0][2]]], 1)
    ccp1obj = SkyCoord(cencoordsp1[0][0], cencoordsp1[0][1], unit='deg')
    print(center.separation(ccp1obj))
    vertscale = center.separation(ccp1obj) * cosmo.kpc_proper_per_arcmin(redshift)
    numpix = int(np.ceil(regionextent.to(u.kpc)/vertscale.to(u.kpc)))
    hzoffset = np.ceil(regionwidth/2)
    wv1 = cube.wavelength[closest(velarr,-velocitylimit)]
    wv2 = cube.wavelength[closest(velarr,velocitylimit)]
    specdata = []
    offsetarr = np.arange(-numpix,numpix+1)
    for i,ii in enumerate(offsetarr):
        thesepixels=[[int(cp[0][1]+off),int(cp[0][0]+ii)] for off in
                     np.arange(-hzoffset,hzoffset)]
        thisspec = extract_spectrum(cube,thesepixels,wvslice=[wv1,wv2])
        specdata.append(thisspec)
    kpcoffset = offsetarr*vertscale.to(u.kpc).value
    veloffset = veltrans(redshift,specdata[0].wavelength.value,restwave)
    pvarr = np.array([sd.flux for sd in specdata])
    return kpcoffset,veloffset,pvarr

def signif_cube(fluxcube, varcube,wave1,wave2,outfile=None):
    nbhdu, errhdu = kt.narrowband(fluxcube,wave1,wave2,varcube=varcube)
    signifdat = nbhdu[0].data/np.sqrt(errhdu[0].data)
    newhdu = nbhdu.copy()
    newhdu[0].data = signifdat
    if outfile is not None:
        fits.writeto(data=newhdu[0].data, header=newhdu[0].header,
                     filename=outfile, overwrite=True)
        #newhdu.writeto(outfile,overwrite=True)
    return newhdu



