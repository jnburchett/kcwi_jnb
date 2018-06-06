import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import contVPmeasure as cvp



from kcwi_jnb import transform as kt

def continuum_subtract(cubefile, varcube, wave1, wave2, outfile=None):
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
                dat[:, idx0, idx1] = thisspec - cont
            except:
                dat[:, idx0, idx1] = 0

    slicedhdu[0].data = dat
    if outfile is not None:
        slicedhdu.writeto(outfile, overwrite=True)
    return slicedhdu


def median_continuum_subtract(cubefile, contwave1, contwave2,
                              linewave1, linewave2, outfile=None):
    """Find continuum by median flux value and subtract from line regiom
    """

    medhdu = kt.narrowband(cubefile, contwave1, contwave2, mode='median')
    linehdu = kt.slice_cube(cubefile, linewave1, linewave2)

    wcs = WCS(medhdu[0].header)
    dat = linehdu[0].data - medhdu[0].data
    sumdat = np.sum(dat,axis=0)

    medhdu[0].data = sumdat
    if outfile is not None:
        medhdu.writeto(outfile, overwrite=True)
    return medhdu


def extract_spectrum(cubefile,pixels,wvslice=None):

    from linetools.spectra.xspectrum1d import XSpectrum1D

    if wvslice is not None:
        thishdulist = kt.slice_cube(cubefile,wvslice[0],wvslice[1])
        dat = thishdulist[0].data
        hdr = thishdulist[0].header
    else:
        dat, hdr = fits.getdata(cubefile, header=True)

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
    wcs = WCS(hdr)
    dim = np.shape(dat)
    wdelt = hdr['PC3_3']
    lastwaveidx = dim[0] - 1
    w1 = wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
    w2 = wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
    newwavearr = np.arange(w1, w2, wdelt)

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



