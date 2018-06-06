import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import NDData
from astropy.nddata import Cutout2D
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


def align_cubes(filelist, obj_align,interp_method='nearest'):
    ra_obj = obj_align.ra.deg
    dec_obj = obj_align.dec.deg
    wobjs = []
    cenpixs = []
    interps = []
    for i, ff in enumerate(filelist):
        dat, hdr = fits.getdata(ff, header=True)
        if i == 0:
            cubedim = np.shape(dat)
            inp = (np.arange(cubedim[0]), np.arange(cubedim[1]),
                   np.arange(cubedim[2]))
        thisw = WCS(hdr)
        wobjs.append(thisw)
        cp = thisw.wcs_world2pix([[ra_obj, dec_obj, 1]], 1)  # (ra,dec,lambda)
        cenpixs.append(cp[0])

        thisinterp = RegularGridInterpolator(inp, dat, method=interp_method,
                                             bounds_error=False)
        interps.append(thisinterp)

    ### Must now figure out dimensions of new cube!
    cenpixs = np.array(cenpixs)
    xmin = np.min(cenpixs[:, 0])
    xmax = np.max(cenpixs[:, 0])
    ymin = np.min(cenpixs[:, 1])
    ymax = np.max(cenpixs[:, 1])
    xdiff = np.ceil(xmax - xmin)
    ydiff = np.ceil(ymax - ymin)

    ### Generate new data cubes with bigger sizes but that will align
    newcubedims = (cubedim[0], int(np.ceil(cubedim[1] + xdiff)), int(np.ceil(cubedim[2] + ydiff)))
    # Modify original header to put object coords in center
    newhdr = hdr.copy()
    newhdr['CRVAL1'] = ra_obj
    newhdr['CRVAL2'] = dec_obj
    newhdr['CRPIX1'] = (newcubedims[2] - 1) / 2.
    newhdr['CRPIX2'] = (newcubedims[1] - 1) / 2.
    newwcs = WCS(newhdr)
    newcubes = []
    # Find coordinates of each pixel in newcube
    idxs = (np.arange(newcubedims[0]), np.arange(newcubedims[1]),
            np.arange(newcubedims[2]))
    idxgrid = np.meshgrid(idxs[0], idxs[1], idxs[2])
    coords = newwcs.wcs_pix2world(idxgrid[0], idxgrid[1], idxgrid[2], 1)

    #pixs = newwcs.wcs_world2pix(coords[0], coords[1], coords[2], 1)
    #pixs = np.array(pixs)

    for i, terp in enumerate(interps):
        # What pixels in the old cube would correspond to the new cubes coords?
        thiscubepix_newcoords = wobjs[i].wcs_world2pix(coords[0], coords[1], coords[2], 1)
        thiscubepix_newcoords = np.array(thiscubepix_newcoords)
        # What would the new pixel values and flux in the single cube be?
        newflux = terp(thiscubepix_newcoords.T)
        nf_trans = np.transpose(newflux, axes=[1, 2, 0])
        newcubes.append(nf_trans)

    return newcubes, newwcs


def change_coord_ref(wcsobj_mod,refcoords,modcoords):
    """Change reference pixels in WCS obj to align with object in different WCS

    Parameters
    ----------
    wcsobj_mod : WCS
        WCS object to be modified
    refcoords : SkyCoord
        Coordinates of reference object in image to be modified
    modcoords : SkyCoord
        Coordinates of reference object in reference image

    Returns
    -------
    wcsobj_mod: WCS
        New WCS transformed to put the reference object in the reference frame

    """
    modpix = modcoords.to_pixel(wcsobj_mod)
    wcsobj_mod.wcs.crval = (refcoords.ra.deg,refcoords.dec.deg)
    wcsobj_mod.wcs.crpix = modpix
    return wcsobj_mod

def change_coord_alignmax(wcsobj_ref,wcsobj_mod,refimage,modimage):
    """Change reference pixels in WCS obj to align brightest pixels in
    different WCS

    Parameters
    ----------
    wcsobj_ref : WCS
        WCS object of refimage
    wcsobj_mod : WCS
        WCS object to be modified
    refimage : 2d array
        Image data for frame that will serve as reference
    modimage : SkyCoord
        Image data for frame that will be modified as reference

    Returns
    -------
    newwcsobj: WCS
        New WCS transformed to put the reference object in the reference frame

    """
    import copy
    nanidxs = np.where(np.isnan(refimage))
    refcopy = refimage.copy()
    refcopy[nanidxs] = 0

    nanidxs = np.where(np.isnan(modimage))
    modcopy = modimage.copy()
    modcopy[nanidxs] = 0

    maxref = np.max(refcopy)
    maxrefpix = np.where(refcopy==maxref)

    ### Indices are returned in Y,X order and are zero indexed where pixels
    ### start with 1 in the FITS standard
    maxrefcoords = wcsobj_ref.wcs_pix2world(maxrefpix[1][0]+1,maxrefpix[0][0]+1,1)
    maxrefcoords = SkyCoord(maxrefcoords[0],maxrefcoords[1],unit='deg')

    ### Range hard-coded here to search for bright pixels due to KCWI edges
    maxmod = np.max(modcopy[16:67,:])
    maxmodpix = np.where(modcopy == maxmod)

    ### Create new WCS object where brightest pixels line up with ref coords
    newwcsobj = copy.deepcopy(wcsobj_mod)
    try:
        newwcsobj.wcs.crval = (maxrefcoords.ra.deg,maxrefcoords.dec.deg)
    except:
        import pdb; pdb.set_trace()
    newwcsobj.wcs.crpix = (maxmodpix[1][0]+1,maxmodpix[0][0]+1)
    return newwcsobj

def cutout(imdata,center,size=[20.*u.arcsec,16.*u.arcsec],wcs=None):
    if isinstance(imdata,str):
        dat,hdr = fits.getdata(imdata,header=True)
        wcs = WCS(hdr)
    elif wcs is None:
        raise IOError("If imdata is an array, wcs must be supplied")
    else:
        dat = imdata
        wcs = wcs

    cutout = Cutout2D(dat,center,size=size,wcs = wcs,
                      copy=True)
    return cutout

def smooth_gauss(imdat,seeing,pixscale):
    """Smooth image data with 2D Gaussian

    Parameters
    ----------
    imdat : 2d array
        Data to be smoothed
    seeing : float
        FWHM of Gaussian kernel (in arcsec)
    pixscale : float
        Pixel scale input data (in arcsec per pixel)

    Returns
    -------


    """
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve_fft

    numseepix = seeing/pixscale
    kern= Gaussian2DKernel(numseepix/2.355)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    convdat = convolve_fft(imdat, kern)
    return convdat


def get_wave_arr(dat,hdr):
    wcs = WCS(hdr)
    try:
        wdelt = hdr['PC3_3']
    except:
        wdelt = hdr['CD3_3']
    dim = np.shape(dat)
    lastwaveidx = np.max(dim) - 1
    w1 = wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
    w2 = wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
    newwavearr = np.arange(w1, w2 + wdelt, wdelt)
    return newwavearr


def slice_cube(cubefile, wave1, wave2, outfile=None):
    """Extract portion of cube extracted in wavelength space.

    Parameters
    ----------
    cubefile
    wave1
    wave2
    outfile

    Returns
    -------

    """
    dat, hdr = fits.getdata(cubefile, header=True)
    wcs = WCS(hdr)
    newwavearr = get_wave_arr(dat,hdr)
    tokeep = np.where((newwavearr >= wave1) & (newwavearr <= wave2))[0]
    try:
        wslice = slice(tokeep[0], tokeep[-1] + 1)
    except:
        import pdb; pdb.set_trace()
    sl2 = slice(0, np.shape(dat)[1])
    sl3 = slice(0, np.shape(dat)[2])

    datslice = dat[tokeep, :, :]
    wcsslice = wcs.slice((wslice, sl2, sl3))

    newhdulist = wcsslice.to_fits()
    newhdulist[0].data = datslice
    if outfile is not None:
        newhdulist.writeto(outfile, overwrite=True)
    return newhdulist

def narrowband(cubefile,wave1,wave2,outfile=None,mode='sum'):
    slicedhdu = slice_cube(cubefile,wave1,wave2)
    dat = slicedhdu[0].data
    if mode == 'sum':
        imgdat = np.sum(dat,axis=0)
    elif mode == 'median':
        imgdat = np.median(dat, axis=0)
    else:
        raise ValueError('Unknown combine type.')
    slicedhdu[0].data=imgdat
    slicedhdu[0].header.remove('CRVAL3')
    slicedhdu[0].header.remove('CRPIX3')
    slicedhdu[0].header.remove('CDELT3')
    slicedhdu[0].header.remove('CUNIT3')
    slicedhdu[0].header.remove('CTYPE3')

    #slicedhdu[0].header.remove('CNAME3')
    slicedhdu[0].header.remove('PC3_3')
    slicedhdu[0].header.set('NAXIS',2)
    slicedhdu[0].header.set('WCSAXES',2)
    slicedhdu.writeto(outfile,overwrite=True)
    return slicedhdu

def apply_filter(cubefile,filter_curve,outfile=None,mode='sum'):
    from scipy.interpolate import interp1d
    from astropy.table import Table
    # Load filter response curve
    filtcurve = Table.read(filter_curve,names=['wave','throughput'],format='ascii')
    fc_interp = interp1d(filtcurve['wave'],filtcurve['throughput'])
    # Slice cube and get wavelength array
    slicedhdu = slice_cube(cubefile,filtcurve['wave'][0],filtcurve['wave'][-1])
    cubedat = slicedhdu[0].data
    newwavearr = get_wave_arr(slicedhdu[0].data,slicedhdu[0].header)
    tpwave = fc_interp(newwavearr)
    transdat_tp = cubedat.T * tpwave
    imgdat = np.sum(transdat_tp.T,axis=0)
    slicedhdu[0].data=imgdat
    slicedhdu[0].header.remove('CRVAL3')
    slicedhdu[0].header.remove('CRPIX3')
    slicedhdu[0].header.remove('CDELT3')
    slicedhdu[0].header.remove('CUNIT3')
    slicedhdu[0].header.remove('CTYPE3')

    slicedhdu[0].header.remove('CNAME3')
    slicedhdu[0].header.remove('PC3_3')
    slicedhdu[0].header.set('NAXIS',2)
    slicedhdu[0].header.set('WCSAXES',2)
    slicedhdu.writeto(outfile,overwrite=True)
    return slicedhdu

def subtract_sky(scicube,varcube,suffix = 'sky.fits'):
    # Collapse into narrowband
    buff = 200
    dat, hdr = fits.getdata(scicube, header=True)
    partdat  = dat[buff:-buff,:,:]
    sumdat = np.sum(partdat,axis=0)
    med = np.median(sumdat[(~np.isnan(sumdat)) & (sumdat > 0)])
    stddev = np.std(sumdat[(~np.isnan(sumdat))&(sumdat>0)])
    notbright = np.where((sumdat<(med+3.*stddev))&(~np.isnan(sumdat))&(sumdat>0))
    sky = np.median(dat[:,notbright[0],notbright[1]],axis=1)
    subtrans = dat.transpose() - sky
    sub = subtrans.transpose()
    sub[np.isnan(sub)] = 0
    wcs = WCS(hdr)
    newhdulist=  wcs.to_fits()
    newhdulist[0].data=sub
    newhdulist.writeto(scicube.split('/')[-1][:-6]+suffix,overwrite=True)


    var, varhdr = fits.getdata(varcube, header=True)
    varskytrans = var.transpose()+sky**2
    varsky = varskytrans.transpose()
    wcs = WCS(varhdr)
    newhdulist = wcs.to_fits()
    newhdulist[0].data = varsky
    newhdulist.writeto(varcube.split('/')[-1][:-6] + suffix, overwrite=True)

    return sub,varsky

def write_fits(imgdata,wcs,outfile,overwrite=True):
    """Write out FITS image with WCS information.

    Parameters
    ----------
    imgdata : image array
        Image data to be written
    wcs : WCS
        WCS of image
    outfile : str
        File name to write to
    overwrite : bool,optional
        If True, overwrite file if one having same name already exists

    Returns
    -------

    """
    newhdulist = wcs.to_fits()
    newhdulist[0].data = imgdata
    newhdulist.writeto(outfile,overwrite=True)

