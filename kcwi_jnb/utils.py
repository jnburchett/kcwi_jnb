import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord

def load_file(inpfile):
    dat, hdr = fits.getdata(inpfile, header=True)
    wcs = WCS(hdr)
    return dat,wcs

def get_wave_arr(dat,hdr,extract_wcs=True):
    if extract_wcs is True:
        wcs = WCS(hdr)
    else:
        wcs = hdr
    try:
        wdelt = wcs.wcs.pc[-1,-1]
        #wdelt = hdr['PC3_3']
    except:
        wdelt = wcs.wcs.cd[-1, -1]
        #wdelt = hdr['CD3_3']
    dim = np.shape(dat)
    lastwaveidx = dim[0] - 1
    w1 = wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
    w2 = wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]

    newwavearr = np.arange(w1, w2+wdelt, wdelt)
    if newwavearr[-1]>w2:
        newwavearr=newwavearr[:-1]
    #if len(newwavearr)!=len(dat)
    return newwavearr

def bright_pix_coords(image,wcs,pixrange=None):
    if pixrange is None:
        maxref = np.max(image)
    else:
        maxref = np.max(image[pixrange[0]:pixrange[1], pixrange[2]:pixrange[3]])
    maxrefpix = np.where(image == maxref)

    ### Indices are returned in Y,X order and are zero indexed where pixels
    ### start with 1 in the FITS standard
    maxrefcoords = wcs.wcs_pix2world(maxrefpix[1][0] + 1, maxrefpix[0][0] + 1, 1)
    maxrefcoords = SkyCoord(maxrefcoords[0], maxrefcoords[1], unit='deg')

    return maxrefpix,maxrefcoords