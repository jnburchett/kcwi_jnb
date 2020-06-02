import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import NDData
from astropy.nddata import Cutout2D
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from kcwi_jnb import utils as ku
from kcwi_jnb.cube import DataCube

def white_light_offset(icubelist,outputdir_nb='nb_off/',outputdir_cubes='corrCubes/',
                       waverange=[4000.,5500.],slicer = 'M', write_aligned_nb=True,
                       cube_prefix='newBrightOffset_',varcubelist=None,
                       pixrange=None,trim=True,nb_subgrad=True):
    import os

    ### make narrowband output directory if necessary
    if not os.path.isdir(outputdir_nb):
        os.makedirs(outputdir_nb)

    if isinstance(icubelist[0],DataCube):
        pass
    else:
        cl = [DataCube(cc) for cc in icubelist]
        icubelist = cl

    ### prep variance cubes if provided
    if varcubelist is not None:
        if isinstance(varcubelist[0], DataCube):
            pass
        else:
            vcl = [DataCube(cc) for cc in varcubelist]
            varcubelist = vcl


    ### choose range of pixels appropriate for slicer

    if slicer in ['M','medium','Medium']:
        if pixrange is None:
            pixrange = [20, 79, 7, 29]
        trimrange = [20, 79, 7, 29]
    elif slicer in ['L','large','Large']:
        if pixrange is None:
            pixrange = [16, 80, 4, 25]
        trimrange = [16, 80, 4, 25]

    ### trim cubes if desired
    if trim is True:
        tricubelist = []
        trvcubelist = []
        for i,cc in enumerate(icubelist):
            tcube = trim_cube(cc, trimrange[0]+1,trimrange[1]+1,trimrange[2]+1,trimrange[3]+1)
            vcube = trim_cube(varcubelist[i], trimrange[0]+1,trimrange[1]+1,trimrange[2]+1,trimrange[3]+1)
            tricubelist.append(tcube)
            trvcubelist.append(vcube)
        icubelist = tricubelist
        varcubelist = trvcubelist

    ### create narrowband images
    nbimages = []
    for cube in icubelist:
        nbout = outputdir_nb + 'nb_' + cube.filename.split('/')[-1]
        nb = narrowband(cube, waverange[0], waverange[1],
                        outfile=nbout)
        if nb_subgrad:
            nbsubdat,nbsubwcs = subtract_gradient(nbout,degree=1,trim=None,outfile=nbout)
            nbsub = DataCube(nbout)
            nbimages.append(nbsub)
        else:
            nbimages.append(nb)
    ### pick one of the narrowband images to use as reference
    refnb = nbimages[0]
    refdat = refnb.data
    refwcs = WCS(refnb.header)

    ### perform the offset
    for i, cube in enumerate(icubelist):
        thisnbw = WCS(nbimages[i].header)
        thisnbdat = nbimages[i].data
        neww = change_coord_alignmax(refwcs, thisnbw, refdat,
                                               thisnbdat,
                                               range_mod=pixrange,
                                               range_ref=pixrange)
        cube.wcs.wcs.crval[0] = neww.wcs.crval[0]
        cube.wcs.wcs.crval[1] = neww.wcs.crval[1]
        cube.wcs.wcs.crpix[0] = neww.wcs.crpix[0]
        cube.wcs.wcs.crpix[1] = neww.wcs.crpix[1]
        if not os.path.isdir(outputdir_cubes):
            os.makedirs(outputdir_cubes)
        cube.write(outputdir_cubes+cube_prefix+cube.filename.split('/')[-1])
        ### offset variance cube if provided
        if varcubelist is not None:
            vcube = varcubelist[i]
            vcube.wcs.wcs.crval[0] = neww.wcs.crval[0]
            vcube.wcs.wcs.crval[1] = neww.wcs.crval[1]
            vcube.wcs.wcs.crpix[0] = neww.wcs.crpix[0]
            vcube.wcs.wcs.crpix[1] = neww.wcs.crpix[1]
            vcube.write(outputdir_cubes+cube_prefix+vcube.filename.split('/')[-1])
        ### if desired, write out white light images from corrected cubes
        if write_aligned_nb:
            aligned_nb_outdir = outputdir_cubes+'nb/'
            if not os.path.isdir(aligned_nb_outdir):
                os.makedirs(aligned_nb_outdir)
            narrowband(cube, waverange[0], waverange[1],
                                 outfile=aligned_nb_outdir+'nb_' + cube.filename.split('/')[-1])
        ##########
    if varcubelist is not None:
        for i, cube in enumerate(varcubelist):
            try:
                thisnbw = WCS(nbimages[i].header)
            except:
                import pdb; pdb.set_trace()
            thisnbdat = nbimages[i].data
            neww = change_coord_alignmax(refwcs, thisnbw, refdat,
                                         thisnbdat,
                                         range_mod=pixrange,
                                         range_ref=pixrange)
            cube.wcs.wcs.crval[0] = neww.wcs.crval[0]
            cube.wcs.wcs.crval[1] = neww.wcs.crval[1]
            cube.wcs.wcs.crpix[0] = neww.wcs.crpix[0]
            cube.wcs.wcs.crpix[1] = neww.wcs.crpix[1]
            if not os.path.isdir(outputdir_cubes):
                os.makedirs(outputdir_cubes)
            cube.write(outputdir_cubes + cube_prefix + cube.filename.split('/')[-1])
    brightcoords = SkyCoord(neww.wcs.crval[0],neww.wcs.crval[1],unit='deg')
    return brightcoords


def align_cubes(cubelist, obj_align,interp_method='nearest'):
    ra_obj = obj_align.ra.deg
    dec_obj = obj_align.dec.deg
    wobjs = []
    cenpixs = []
    interps = []
    for i, cube in enumerate(cubelist):
        if isinstance(cube, str):
            cube = DataCube(cube)
            dat = cube.data
            thisw = cube.wcs
        else:
            dat = cube.data
            thisw = cube.wcs

        if i == 0:
            cubedim = np.shape(dat)
            print(cubedim)
            inp = (np.arange(cubedim[0]), np.arange(cubedim[1]),
                   np.arange(cubedim[2]))
        wobjs.append(thisw)
        cp = thisw.wcs_world2pix([[ra_obj, dec_obj, 1]], 1)  # (ra,dec,lambda)
        cenpixs.append(cp[0])
        try:
            thisinterp = RegularGridInterpolator((inp[0],inp[1],inp[2]), dat, method=interp_method,
                                             bounds_error=False)
        except:
            import pdb; pdb.set_trace()
        interps.append(thisinterp)

    ### Must now figure out dimensions of new cube!
    cenpixs = np.array(cenpixs)
    try:
        xmin = np.min(cenpixs[:, 0])
    except:
        import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    xmax = np.max(cenpixs[:, 0])
    ymin = np.min(cenpixs[:, 1])
    ymax = np.max(cenpixs[:, 1])
    xdiff = np.ceil(xmax - xmin)
    ydiff = np.ceil(ymax - ymin)

    ### Generate new data cubes with bigger sizes but that will align
    newcubedims = (cubedim[0], int(cubedim[1] + xdiff), int((cubedim[2] + ydiff)))
    # Modify original header to put object coords in center
    newhdr = cube.header.copy()
    newhdr['CRVAL1'] = ra_obj
    newhdr['CRVAL2'] = dec_obj
    newhdr['CRPIX1'] = (newcubedims[2] + 1.) / 2.
    newhdr['CRPIX2'] = (newcubedims[1] + 1.) / 2.
    newwcs = WCS(newhdr)

    newcubes = []
    # Find coordinates of each pixel in newcube
    idxs = (np.arange(newcubedims[0]), np.arange(newcubedims[1]),
            np.arange(newcubedims[2]))
    idxgrid = np.meshgrid(idxs[0], idxs[1], idxs[2],indexing='ij')
    newcoords = newwcs.wcs_pix2world(idxgrid[2]+1, idxgrid[1]+1, idxgrid[0], 1)

    #pixs = newwcs.wcs_world2pix(coords[0], coords[1], coords[2], 1)
    #pixs = np.array(pixs)

    for i, terp in enumerate(interps):
        # What pixels in the old cube would correspond to the new cubes coords?
        thiscubepix_newcoords = wobjs[i].wcs_world2pix(
            newcoords[0], newcoords[1], newcoords[2], 0)
        thiscubepix_newcoords = np.array(thiscubepix_newcoords)
        tcp_nc_x, tcp_nc_y, tcp_nc_wave = thiscubepix_newcoords
        # What would the new pixel values and flux in the single cube be?
        newflux = terp((tcp_nc_wave, tcp_nc_y, tcp_nc_x))
        newcubes.append(newflux)

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

def change_coord_alignmax(wcsobj_ref,wcsobj_mod,refimage,modimage,
                          range_ref=None,range_mod=None):
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
    range_ref, range_mod : 4-element list or tuple, optional
        Bottom, top, left, right of box within which to search for max values

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

    if range_ref is None:   
        maxref = np.max(refcopy)
    else:
        try:
            maxref = np.max(refcopy[range_ref[0]:range_ref[1],range_ref[2]:range_ref[3]])
        except:
            import pdb; pdb.set_trace()
    maxrefpix = np.where(refcopy==maxref)

    ### Indices are returned in Y,X order and are zero indexed where pixels
    ### start with 1 in the FITS standard
    maxrefcoords = wcsobj_ref.wcs_pix2world(maxrefpix[1][0]+1,maxrefpix[0][0]+1,1)
    maxrefcoords = SkyCoord(maxrefcoords[0],maxrefcoords[1],unit='deg')

    ### Range hard-coded here to search for bright pixels due to KCWI edges
    if range_mod is None:
        maxmod = np.max(modcopy)
    else:
        maxmod = np.max(modcopy[range_mod[0]:range_mod[1],range_mod[2]:range_mod[3]])
    maxmodpix = np.where(modcopy == maxmod)
    print(maxmodpix)

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

def smooth_gauss(imdat,fwhm,norm=1.,angle=None,pixscale=0.0273):
    """Smooth image data with 2D Gaussian

    Parameters
    ----------
    imdat : 2d array
        Data to be smoothed
    fwhm : float or 2-element list
        FWHM of Gaussian kernel (in arcsec)
    pixscale : float
        Pixel scale input data (in arcsec per pixel)

    Returns
    -------


    """
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve_fft

    if isinstance(fwhm,float):
        numseepix = fwhm / pixscale
        kern = Gaussian2DKernel(numseepix / 2.355)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    else:
        try:
            fwhmx, fwhmy = fwhm
        except:
            import pdb;pdb.set_trace()
        numseepix_x = fwhmx / pixscale
        numseepix_y = fwhmy / pixscale
        if angle is None:
            angle = 0
        try:
            kern = Gaussian2DKernel(numseepix_x / 2.355, numseepix_y / 2.355,theta=angle)
        except:
            import pdb;pdb.set_trace()

    convdat = norm * convolve_fft(imdat, kern)
    return convdat


def slice_cube(cube, wave1, wave2, outfile=None):
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

    if isinstance(cube,str):
        cube = DataCube(cube)
        dat = cube.data
        wcs = cube.wcs
    else:
        dat = cube.data
        wcs = cube.wcs
    newwavearr = cube.wavelength
    if np.all(newwavearr<1e-3):
        newwavearr *= 1e10

    tokeep = np.where((newwavearr >= wave1) & (newwavearr <= wave2))[0]
    wslice = slice(tokeep[0], tokeep[-1]+1)

    sl2 = slice(0, np.shape(dat)[1])
    sl3 = slice(0, np.shape(dat)[2])

    datslice = dat[tokeep, :, :]
    try:
        wcsslice = wcs.slice((wslice, sl2, sl3))
    except:
        import pdb; pdb.set_trace()

    newhdulist = wcsslice.to_fits()
    newhdulist[0].data = datslice
    if outfile is not None:
        newhdulist.writeto(outfile, overwrite=True)
    return DataCube(newhdulist)

def trim3dwcs(wcs):
    newwcs = WCS(naxis=2)
    try:
        #wcs.wcs.naxis = 2
        newwcs.wcs.crval = wcs.wcs.crval[:-1]
    except:
        import pdb; pdb.set_trace()
    newwcs.wcs.crpix = wcs.wcs.crpix[:-1]
    newwcs.wcs.cdelt = wcs.wcs.cdelt[:-1]
    newwcs.wcs.cunit= [wcs.wcs.cunit[0],wcs.wcs.cunit[1]]
    newwcs.wcs.ctype = [wcs.wcs.ctype[0],wcs.wcs.ctype[1]]
    newwcs.wcs.cname = [wcs.wcs.cname[0],wcs.wcs.cname[1]]
    newwcs.wcs.pc = wcs.wcs.pc[:-1,:-1]

    #for kk in ['CRVAL3','CRPIX3','CDELT3','CUNIT3','CTYPE3','CNAME3','PC3_3']:
    #    if kk in header.keys():
    #        header.remove(kk)
    #header.set('NAXIS',2)
    #header.set('WCSAXES',2)
    return newwcs
    

def narrowband(cube,wave1,wave2,outfile=None,mode='sum',varcube=None):
    from astropy.io import fits

    slicedhdu = slice_cube(cube,wave1,wave2)
    dat = slicedhdu.data

    if mode == 'sum':
        imgdat = np.sum(dat,axis=0)
    elif mode == 'median':
        imgdat = np.median(dat, axis=0)
    else:
        raise ValueError('Unknown combine type.')
    slicedhdu.data=imgdat
    trimwcs = trim3dwcs(WCS(slicedhdu.header))
    slicedhdu.wcs = trimwcs
    slicedhdu.header = trimwcs.to_header()
    if outfile is not None:
        fits.writeto(data=slicedhdu.data,header=slicedhdu.header,
                     filename=outfile,overwrite=True)
    if varcube is None:
        return slicedhdu
    else:
        if outfile is not None:
            erroutfile = outfile[:-5]+'_err.fits'
            errhdu = narrowband_err(varcube,wave1,wave2,outfile=erroutfile)
        else:
            errhdu = narrowband_err(varcube, wave1, wave2, outfile=outfile)
        return slicedhdu,errhdu


def narrowband_err(varcube,wave1,wave2,outfile=None,mode='sum'):
    from astropy.io import fits
    slicedhdu = slice_cube(varcube,wave1,wave2)
    dat = slicedhdu.data
    if mode == 'sum':
        imgdat = np.sum(dat,axis=0)
    elif mode == 'median':
        imgdat = np.median(dat, axis=0)
    else:
        raise ValueError('Unknown combine type.')
    slicedhdu.data=imgdat
    slicedhdu.header = trim3dwcs(WCS(slicedhdu.header)).to_header()
    if outfile is not None:
        fits.writeto(data=slicedhdu.data,header=slicedhdu.header,
                     filename=outfile,overwrite=True)
    return slicedhdu


def apply_filter(cubefile,filter_curve,outfile=None,mode='sum'):
    from scipy.interpolate import interp1d
    from astropy.table import Table
    # Load filter response curve
    filtcurve = Table.read(filter_curve,names=['wave','throughput'],format='ascii')
    fc_interp = interp1d(filtcurve['wave'],filtcurve['throughput'])
    # Slice cube and get wavelength array
    slicedhdu = slice_cube(cubefile,filtcurve['wave'][0],filtcurve['wave'][-1])
    cubedat = slicedhdu.data
    newwavearr = ku.get_wave_arr(slicedhdu.data,slicedhdu.header)
    tpwave = fc_interp(newwavearr)
    transdat_tp = cubedat.T * tpwave
    imgdat = np.sum(transdat_tp.T,axis=0)
    slicedhdu.data=imgdat
    slicedhdu.header.remove('CRVAL3')
    slicedhdu.header.remove('CRPIX3')
    slicedhdu.header.remove('CDELT3')
    slicedhdu.header.remove('CUNIT3')
    slicedhdu.header.remove('CTYPE3')

    slicedhdu.header.remove('CNAME3')
    slicedhdu.header.remove('PC3_3')
    slicedhdu.header.set('NAXIS',2)
    slicedhdu.header.set('WCSAXES',2)
    if outfile is not None:
        slicedhdu.write(outfile,overwrite=True)
    return slicedhdu

def subtract_sky(scicube,varcube,suffix = 'skysub.fits',return_sky=False,
                 skysuffix='sky.fits'):
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
    newhdulist.writeto(scicube.split('/')[-1][:-5]+suffix,overwrite=True)


    var, varhdr = fits.getdata(varcube, header=True)
    varsky = var
    wcs = WCS(varhdr)
    newhdulist = wcs.to_fits()
    newhdulist[0].data = varsky
    newhdulist.writeto(varcube.split('/')[-1][:-5] + suffix, overwrite=True)

    if return_sky:
        newhdulist = wcs.to_fits()
        newhdulist[0].data = sky.transpose()
        newhdulist.writeto(scicube.split('/')[-1][:-5] + skysuffix, overwrite=True)
        return sub,varsky,sky
    else:
        return sub,varsky

def subtract_master_sky(scicubes,mastersky,outdir = None,suffix='mskysub.fits'):

    if type(scicubes) == list:
        for i,sc in enumerate(scicubes):
            if not isinstance(sc,DataCube):
                cube = DataCube(sc)
                rootfn = sc.split('/')[-1][:-5]
            cubedatsub = cube.data.transpose() - mastersky
            cube.data = cubedatsub.transpose()
            if outdir is not None:
                cube.write(outdir+rootfn+'_'+suffix)
    else:
        if not isinstance(scicubes, DataCube):
            cube = DataCube(scicubes)
            rootfn = scicubes.split('/')[-1][:-5]
        else:
            cube = scicubes
            rootfn = ''
        cubedatsub = cube.data.transpose() - mastersky
        cube.data = cubedatsub.transpose()
        if outdir is not None:
            cube.write(outdir + rootfn + '_' + suffix)
        return cube


def subtract_gradient(nbimage, trim = [11,70,6,28], nonbgregion=[22,38,2,17],
                      outfile=None,return_gradModel=False,degree=2,floor=None):
    """Subtract residual gradient of background from narrowband image
    
    Parameters
    ----------
    nbimage : str
        Image from which to subtract background
    trim : 4-element list, optional
        Indices of pixels in image to keep
    nonbgregion : 4-element list, optional
        Indices of pixels in image that are not to be used to assess background
    outfile : str
        Filename for output image

    Returns
    -------
    subimage : image array
        Image data with background subtracted

    """

    from astropy.modeling import models, fitting
    import warnings

    dat, hdr = fits.getdata(nbimage, header=True)
    wcs = WCS(hdr)
    if trim is not None:
        dat = dat[trim[0]:trim[1],trim[2]:trim[3]]
        wcs.wcs.crpix -= np.array([trim[2], trim[0]])
    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=degree)
    fit_p = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:np.shape(dat)[0], :np.shape(dat)[1]]
    if nonbgregion is not None:
        tofit = np.ma.masked_array(dat, mask=np.zeros_like(dat))
        tofit[nonbgregion[0]:nonbgregion[1],nonbgregion[2]:nonbgregion[3]].mask = True
    else:
        tofit = dat
    #with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
     #   warnings.simplefilter('ignore')
    try:
        p = fit_p(p_init, x, y, tofit)
    except:
        import pdb; pdb.set_trace()
    if floor is None:
        dat -= p(x, y)
    elif floor=='min':
        dat -= (p(x, y)-np.min(p(x,y)))
    else:
        dat -= (p(x, y)-floor)
    
    if outfile is not None:
        newtrim = wcs.to_fits()
        newtrim[0].data = dat
        newtrim.writeto(outfile,overwrite=True)
    if return_gradModel is True:
        return dat, wcs, p(x,y)
    else:
        return dat, wcs










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

def convolve_cube(cube,convpars=[1.,1.56767728,1.98803262,2.31844267],
                  pixscale=0.2):
    norm, fwhm_1, fwhm_2, theta = convpars
    for i in range(np.shape(cube.data)[0]):
        cube.data[i] = smooth_gauss(cube.data[i],(fwhm_1, fwhm_2),norm,
                                    angle=theta,pixscale=pixscale)
    return cube

def convolve_reproject_cube(cube,basecube,pixscale=0.2,bounds=None,
                            convpars=[1.,1.56767728,1.98803262,2.31844267],
                            PA=27):
    from kcwi_jnb import model as km
    from importlib import reload
    reload(km)

    norm, fwhm_1, fwhm_2, theta = convpars
    trimmedbasewcs = trim3dwcs(basecube.wcs)
    trimmedcubewcs = trim3dwcs(cube.wcs)
    newcubedata = []
    for i in range(np.shape(cube.data)[0]):
        newcubedata.append(km.convolve_reproj(cube.data[i],norm, (fwhm_1, fwhm_2),
                                          basecube.data[0],trimmedcubewcs,trimmedbasewcs,
                                          angle=theta, pixscale=pixscale,bounds=None))
    newwcs = basecube.wcs.deepcopy()
    newwcs.wcs.crpix[2] = cube.wcs.wcs.crpix[2]
    newwcs.wcs.crval[2] = cube.wcs.wcs.crval[2]
    newwcs.wcs.cdelt[2] = cube.wcs.wcs.cdelt[2]
    newwcs.wcs.pc[2,2] = cube.wcs.wcs.pc[2,2]

    ydelt = 0.2915 / 3600.
    xdelt = 0.6789 / 3600.
    newwcs.wcs.cdelt[0] = 1.
    newwcs.wcs.cdelt[1] = 1.
    rotangle = np.radians(PA)  # Assuming PA given in degress
    psmat = np.array([[xdelt * np.cos(rotangle), -ydelt * np.sin(rotangle), 0.],
                      [xdelt * np.sin(rotangle), ydelt * np.cos(rotangle), 0.],
                      [0., 0., cube.wcs.wcs.pc[2,2]]])
    newwcs.wcs.cd = psmat
    cube.wcs = newwcs
    cube.data = np.array(newcubedata)
    return cube

def trim_cube(cube,y1=11,y2=70,x1=6,x2=28,outfile=None):
    """Trim spatial dimensions of data cube and adjust WCS accordingly.

    Parameters
    ----------
    cube : DataCube or str
        Data cube to be trimmed
    y1, y2, x1, x2 : int
        Pixels to keep in trimmed cube (1-indexed as might be represented in DS9)
    outfile : str, optional
        File name to write to

    Returns
    -------
    cube : DataCube
        Trimmed data cube
    """
    if isinstance(cube,str):
        cube=DataCube(cube)
    cube.data = cube.data[:,y1-1:y2-1,x1-1:x2-1]
    crpix_old = cube.wcs.wcs.crpix

    cube.wcs.wcs.crpix[0]=crpix_old[0]-(x1-1)
    cube.wcs.wcs.crpix[1] = crpix_old[1]-(y1-1)

    if outfile is not None:
        cube.write(outfile)

    return cube

def trim_cube_relpix(cube, cenpix, dx, dy, zeroindex = True,outfile=None,dims='yxz'):
    """Trim spatial dimensions of data cube relative to some central pixel
    and adjust WCS accordingly.

    Parameters
    ----------
    cube : DataCube or str
        Data cube to be trimmed
    cenpix : tuple
        Central pixel indices as (y,x)
    dx : int
        Number of pixels to keep in the x-direction
    dy : int
        Number of pixels to keep in the y-direction
    zeroindex : bool, optional
        If True, cenpix is true index of central pixel but 0-indexed array
        If False, 1-indexed
    outfile : str, optional
        File name to write to

    Returns
    -------
    cube : DataCube
        Trimmed data cube
    """
    if isinstance(cube,str):
        cube=DataCube(cube)

    if not zeroindex:
        cenpix[0] -= 1.
        cenpix[1] -= 1.

    if dims=='zyx':
        cube.data = cube.data[:,cenpix[0][0]-dy+1:cenpix[0][0]+dy-1,
                    cenpix[1][0]-dx+1:cenpix[1][0]+dx-1]
    elif dims=='yxz':
        cube.data = cube.data[cenpix[0][0] - dy + 1:cenpix[0][0] + dy - 1,
                    cenpix[1][0] - dx + 1:cenpix[1][0] + dx - 1, :]
    elif dims=='yx':
        cube.data = cube.data[cenpix[0][0] - dy + 1:cenpix[0][0] + dy - 1,
                    cenpix[1][0] - dx + 1:cenpix[1][0] + dx - 1]
    crpix_old = cube.wcs.wcs.crpix

    cube.wcs.wcs.crpix[0] = crpix_old[0] - (dx - 1)
    cube.wcs.wcs.crpix[1] = crpix_old[1] - (dy - 1)

    if outfile is not None:
        cube.write(outfile)

    return cube



def convert_wavelength(cube,outfile=None):
    from linetools.spectra.xspectrum1d import XSpectrum1D
    dims = np.shape(cube.data[0])
    lastwaveidx = np.max(dims) - 1
    wave1 = cube.wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
    wave2 = cube.wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
    import pdb; pdb.set_trace()
    if wave1 > 1000.:
        inc = 1.
    else:
        inc = 1e-10
    newwavearr = np.arange(wave1, wave2 + inc, inc)
    newspec = XSpectrum1D(newwavearr * inc, np.zeros_like(newwavearr))
    newspec.meta['airvac'] = 'air'
    newspec.airtovac()
    diffs = newspec.wavelength[1:] - newspec.wavelength[0:-1]

    # TODO: Come up with a way to keep the header info and WCS synced
    cube.wcs.wcs.crval[2] = newspec.wvmin.value
    cube.wcs.wcs.crpix[2] = 1.
    cube.wcs.wcs.pc[2][2] = np.median(diffs.value)
    newhdulist = cube.wcs.to_fits()
    newhdulist[0].data = cube.data

    newhdulist[0].header.set('CTYPE3', 'VAC', 'Vacuum wavelengths')
    newhdulist[0].header['CUNIT3'] = 'Angstrom'
    newhdulist[0].header['CD3_3'] = np.median(diffs.value)
    # cube.header['CRPIX3'] = 1
    # cube.header['PC3_3'] = np.median(diffs.value)
    # cube.header['CRVAL3'] = newspec.wvmin.value

    cube.header = newhdulist[0].header

    #import pdb; pdb.set_trace()
    if outfile is not None:
        newhdulist.writeto(outfile, overwrite=True)



