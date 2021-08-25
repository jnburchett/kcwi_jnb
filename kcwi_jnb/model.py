import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve,convolve_fft
from scipy import optimize
from reproject import reproject_interp
import pdb
from astropy.cosmology import Planck15 as cosmo
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from linetools.spectra.xspectrum1d import XSpectrum1D
from kcwi_jnb import transform
from kcwi_jnb.cube import DataCube
from kcwi_jnb import utils


# Lines below tried a 2-parameter fit...took too long
'''
def fit_seeing(highres,lowres,wcs_hires,wcs_lowres,initpars=np.array([30.,0.7,6.2]),
           pixscale_highres=0.03,fitbounds=[[10,23],[37,52]]):
    bounds = [[1.,0.01,0.],[1000.,4.,10.]]
    sol = optimize.least_squares(err_func,initpars,bounds=bounds,
            args =(highres,lowres,wcs_hires,wcs_lowres,pixscale_highres))
    return sol
'''


def fit_seeing(highres,lowres,wcs_hires,wcs_lowres,init_norm=1.,init_fwhm=1.3,
               init_offset=0.,init_theta=0.,pixscale_highres=0.03,
               fitbounds=[[8,25],[35,54]],parbounds=[[0.1,0.1],[30,4.]],**kwargs):
    """Fit a Gaussian 2D kernel to an HST image using the KCWI at lower resolution

    Parameters
    ----------
    highres : image array
        High-resolution image (from HST)
    lowres : image array
        Low-resolution image (from kcwi_jnb)
    wcs_hires : WCS
        WCS of HST image
    wcs_lowres : WCS
        WCS of KCWI image
    init_norm : float,optional
        Initial guess for the normalization
    init_fwhm : float or 2-element array,optional
        Initial guess for the seeing FWHM; if 2 elements, use asymmetric Gaussian
    init_offset : float,optional
        Initial guess for the background offset of fit
    init_angle : float, optional
        Initial guess for the rotation angle of kernel if asymmetric
    pixscale_highres : float, optional
        Pixel scale of HST image
    fitbounds : list of 2 2-element lists
        Pixel range over which to fit
    parbounds : list of 2 2-element lists
        Bounds on fitted parameters: [[x0_lo,x1_lo],[x0_hi,x1_hi]]

    Returns
    -------
    sol : scipy.optimize.optimze.OptimizeResult
        Solution from scipy.optimize.least_squares()

    """
    if not isinstance(init_fwhm,float):
        ef = err_func_2d
        if init_offset is None:
            initpars = np.array([init_norm, init_fwhm[0], init_fwhm[1],
                                 init_theta])
        else:
            initpars = np.array([init_norm,init_fwhm[0],init_fwhm[1],
                             init_theta,init_offset])
    elif init_offset is None:
        ef = err_func_nooff
        initpars = np.array([init_norm, init_fwhm])
    else:
        ef = err_func
        initpars = np.array([init_norm, init_fwhm,init_offset])
        print(initpars)

    sol = optimize.least_squares(ef,initpars,bounds=parbounds,
            args =(highres,lowres,wcs_hires,wcs_lowres,pixscale_highres,
                   np.array(fitbounds)),**kwargs)

    return sol

def err_func_nooff(pars,inp,data,wcs_hires,wcs_lowres,pixscale=0.03,
             bounds=np.array([[6,20],[42,57]])):
    norm,fwhm = pars
    numseepix = fwhm / pixscale
    nanpix_data = np.where(np.isnan(data))
    data[nanpix_data]=0

    kern = Gaussian2DKernel(numseepix / 2.355)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    convdat = norm*convolve_fft(inp, kern)
    #convdat = norm * convolve(inp, kern, boundary='extend')
    inpreproj, inpfootprint = \
        reproject_interp((convdat, wcs_hires), wcs_lowres, order='nearest-neighbor',
                         shape_out=np.shape(data))
    model = inpreproj
    model[nanpix_data]=0
    nanpix_inp = np.where(np.isnan(model))
    model[nanpix_inp] = 0
    data[nanpix_inp]=0
    return np.sum((data[bounds[1,0]:bounds[1,1],bounds[0,0]:bounds[0,1]] -
                   model[bounds[1, 0]:bounds[1, 1], bounds[0, 0]:bounds[0, 1]])**2)

def err_func(pars,inp,data,wcs_hires,wcs_lowres,pixscale=0.03,
             bounds=np.array([[6,20],[42,57]])):
    norm,fwhm,offset = pars
    numseepix = fwhm / pixscale
    nanpix_data = np.where(np.isnan(data))
    data[nanpix_data]=0

    kern = Gaussian2DKernel(numseepix / 2.355)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    if offset is None:
        convdat = norm*convolve_fft(inp, kern, boundary='extend')
    else:
        convdat = norm * convolve_fft(inp, kern, boundary='extend') + offset
    #convdat = norm * convolve(inp, kern, boundary='extend')
    inpreproj, inpfootprint = \
        reproject_interp((convdat, wcs_hires), wcs_lowres, order='nearest-neighbor',
                         shape_out=np.shape(data))
    model = inpreproj
    model[nanpix_data]=0
    nanpix_inp = np.where(np.isnan(model))
    model[nanpix_inp] = 0
    data[nanpix_inp]=0
    return np.sum((data[bounds[1,0]:bounds[1,1],bounds[0,0]:bounds[0,1]] -
                   model[bounds[1, 0]:bounds[1, 1], bounds[0, 0]:bounds[0, 1]])**2)

def err_func_2d(pars,inp,data,wcs_hires,wcs_lowres,pixscale=0.03,
             bounds=np.array([[6,20],[42,57]])):
    #norm,fwhmx,fwhmy = pars
    if len(pars)==5:
        norm,fwhmx,fwhmy,theta, offset = pars
    else:
        norm, fwhmx, fwhmy, theta = pars
        offset = 0.
    numseepix_x = fwhmx / pixscale
    numseepix_y = fwhmy / pixscale
    nanpix_data = np.where(np.isnan(data))
    data[nanpix_data]=0

    kern = Gaussian2DKernel(numseepix_x / 2.355,numseepix_y / 2.355,
                            theta=theta)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    #convdat = norm*convolve_fft(inp, kern,boundary='wrap')

    convdat = norm * convolve_fft(inp, kern) + offset

    inpreproj, inpfootprint = \
        reproject_interp((convdat, wcs_hires), wcs_lowres, order='bicubic',
                         shape_out=np.shape(data))
    model = inpreproj
    model[nanpix_data]=0
    nanpix_inp = np.where(np.isnan(model))
    model[nanpix_inp] = 0
    data[nanpix_inp]=0
    return np.sum((data[bounds[1,0]:bounds[1,1],bounds[0,0]:bounds[0,1]] -
                   model[bounds[1, 0]:bounds[1, 1], bounds[0, 0]:bounds[0, 1]])**2)



def convolve_reproj(inp,norm,fwhm,data,wcs_hires,wcs_lowres,offset=None,pixscale=0.03,
                    bounds=np.array([[10,23],[37,52]]),inneroffset=None,angle=None):
    """Say what this does

        Parameters
        ----------
        wcs_hires : WCS
            WCS of HST image
        wcs_lowres : WCS
            WCS of KCWI image
        norm : float,optional
            normalization factor; units??
        fwhm : float or 2-element array,optional
            seeing FWHM; if 2 elements, use asymmetric Gaussian; what are units??
            looks like should be arcsec??


        Returns
        -------


        """


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
    if (offset is None)&(inneroffset is None):
        convdat = norm * convolve_fft(inp, kern, boundary='wrap')
    elif (inneroffset is None):
        convdat = norm * convolve_fft(inp, kern, boundary='wrap') + offset
    elif (offset is None):
        convdat = norm * convolve_fft(inp + inneroffset, kern, boundary='wrap')
    else:
        convdat = norm * convolve_fft(inp + inneroffset, kern,
                                      boundary='wrap') + offset
    #convdat = norm * convolve(inp, kern, boundary='extend')
    try:
        inpreproj, inpfootprint = \
            reproject_interp((convdat, wcs_hires), wcs_lowres,
                         order='nearest-neighbor',
                         shape_out=np.shape(data))
    except:
        import pdb; pdb.set_trace()
    if offset is not None:
        inpreproj += offset
    if bounds is not None:
        return inpreproj[bounds[1,0]:bounds[1,1],bounds[0,0]:bounds[0,1]]
    else:
        return inpreproj



def fit_seeing_cf(highres,lowres,wcs_hires,wcs_lowres,initpars=np.array([30.,0.7]),
           pixscale_highres=0.03,bounds=np.array([[10,23],[37,52]])):
    data = lowres[bounds[1,0]:bounds[1,1],bounds[0,0]:bounds[0,1]]
    sigma = np.ones_like(data)
    sol = optimize.curve_fit(convolve_reproj,highres,data,p0=initpars,
                             sigma=sigma,args =(highres,data,wcs_hires,wcs_lowres))
    return sol



def addWcsRtModel(inp,ref,outfile,spatial_scale_xy=[0.679,0.290],zgal=0.6942,
                  center=('12 36 19.811', '+62 12 51.831'),PA=27):

    ### Load in the model file
    inpdat = fits.open(inp)
    rtcube = DataCube(inp,include_wcs=False)
    rtcube.wavelength = inpdat[2].data
    rtcube.data = inpdat[1].data
    wavearr = rtcube.wavelength

    ### Load in the reference file if necessary
    if not isinstance(ref,DataCube):
        refcube = DataCube(ref)
    else:
        refcube = inp
    refdat = refcube.data
    refwave = refcube.wavelength
    nbhdu = transform.narrowband(refcube,4000.,5000.)
    refnb = nbhdu.data
    refwcs = WCS(nbhdu.header)

    ### Rearrange the cube and find brightest pixel
    newdat= np.transpose(rtcube.data, axes=[2, 1, 0])
    sumdat = np.sum(newdat,axis=0)
    maxdat = np.max(sumdat)
    brightmodpix = np.where(sumdat == maxdat)
    print(brightmodpix)


    ### Get the pixel scale in the spectral dimension
    diff = wavearr[1:]-wavearr[:-1]
    if not np.all(np.abs(diff-diff[0])<1e-3):
        import pdb; pdb.set_trace()
        raise ValueError('Wavelength array not uniformly sampled')
    spectral_scale = diff[0]

    ### Set scale and reference position in model
    wave1 = wavearr[0]*(1.+zgal)
    spectral_scale *= (1.+zgal)

    ### Get the bright pixel from reference cube
    brightpix, brightcoords = utils.bright_pix_coords(refnb, refwcs)

    ### Create the new WCS
    newwcs = WCS(naxis=3)

    if center is not None:
        cencoords = SkyCoord(center[0],center[1],unit=(u.hourangle,u.deg))
    else:
        cencoords = brightcoords
    # coordinates are (y, x, lam) and '1' indexed in FITS files
    newwcs.wcs.crpix = np.array([brightmodpix[1][0]+1,brightmodpix[0][0]+1, 1])
    newwcs.wcs.crval = np.array([cencoords.ra.deg, cencoords.dec.deg, wave1])
    newwcs.wcs.cdelt = np.array([spatial_scale_xy[0]/3600., spatial_scale_xy[1]/3600., spectral_scale])
    newwcs.wcs.cunit = ['deg', 'deg', 'Angstrom']
    newwcs.wcs.ctype = ('RA---TAN', 'DEC--TAN', 'VAC')
    rotangle = np.radians(PA) # Assuming PA given in degrees
    psmat = np.array([[newwcs.wcs.cdelt[0] * np.cos(rotangle), -newwcs.wcs.cdelt[1] * np.sin(rotangle), 0.],
                      [newwcs.wcs.cdelt[0] * np.sin(rotangle), newwcs.wcs.cdelt[1] * np.cos(rotangle), 0.],
                      [0., 0., 1.]])
    newwcs.wcs.cd = psmat




    newhdu = newwcs.to_fits()
    newhdu[0].header['PC3_3']=spectral_scale
    newhdu[0].data = newdat

    newhdu.writeto(outfile, overwrite=True)



def rtModelErrFunc(pars, inp, data, wcs_model, wcs_data, pixscale=0.0273,
                   objcoords = SkyCoord('12 36 19.811', '+62 12 51.831',
                                        unit=(u.hourangle,u.deg)),
                   convpars=[50.31491672,1.56767728,1.98803262,2.31844267],
                   fitbounds = [10,30,3,10]):
    ''' Error function for fitting radiative transfer models.  The two
    parameters to fit are the ratio of brightness between the continuum source
    (typically the only pixels with >2 counts) and the next brightest pixels
    from the wind (typically < 0.01).

    Parameters
    ----------
    pars
    inp
    data
    wcs_model
    wcs_data
    pixscale
    bounds
    convpars

    Returns
    -------

    '''

    contlumratio,globalnorm = pars
    convnorm, fwhm_1, fwhm_2, theta =convpars
    inp[np.isnan(inp)] = 0
    maxwindbright = np.max(inp[inp<2.])
    inp[inp>2.] = maxwindbright*contlumratio

    convbounds = None
    convdat = convolve_reproj(globalnorm*inp, convnorm, (fwhm_1, fwhm_2),
                                data, wcs_model, wcs_data, pixscale=pixscale,
                                bounds=convbounds, angle=theta)
    datacut = transform.cutout(data, objcoords, size=[37, 15], wcs=wcs_data)
    modelcut = transform.cutout(convdat, objcoords, size=[37, 15], wcs=wcs_data)
    modelcut.data[np.isnan(modelcut.data)]=0.
    fb1,fb2,fb3,fb4 = fitbounds
    diff = datacut.data[fb1:fb2,fb3:fb4] -  modelcut.data[fb1:fb2,fb3:fb4]
    return np.sum(diff**2)


def rtModelConvCut(contlumratio,globalnorm,inp,data,wcs_model, wcs_data,
                   pixscale=0.0273,
                   objcoords = SkyCoord('12 36 19.811', '+62 12 51.831',
                                        unit=(u.hourangle,u.deg)),
                   convpars=[50.31491672,1.56767728,1.98803262,2.31844267]):
    convnorm, fwhm_1, fwhm_2, theta =convpars
    inp[np.isnan(inp)] = 0
    maxwindbright = np.max(inp[inp<2.])
    inp[inp>2.] = maxwindbright*contlumratio

    convbounds = None
    convdat = convolve_reproj(globalnorm*inp, convnorm, (fwhm_1, fwhm_2),
                                data, wcs_model, wcs_data, pixscale=pixscale,
                                bounds=convbounds, angle=theta)
    datacut = transform.cutout(data, objcoords, size=[37, 15], wcs=wcs_data)
    modelcut = transform.cutout(convdat, objcoords, size=[37, 15], wcs=wcs_data)
    modelcut.data[np.isnan(modelcut.data)]=0.
    return modelcut


def rtModel_spatialconv_rebin(infil, outfil,spatial_scale_xy=[0.679,0.290],zgal=0.6942,\
                              sm_fwhm_arcsec=1.634, outroot=None):
    '''Convolve radiative transfer model cube.  Originally from read_spec_output.read_fullfits_conv_rebin

    Inputs
    ------
    infil : unprocessed RT model fits file (the cube to convolve)

    outroot : str
    Filename (before extension) of output file
    '''

    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(zgal).to(u.kpc / u.arcsec).value
    sm_fwhm = sm_fwhm_arcsec * kpc_per_arcsec  # in kpc

    #sm_fwhm = 7.209      # in kpc  -- 1" arcsec at z=0.6942
    #sm_fwhm = 9.360      # in kpc  -- 1.277" arcsec at z=0.6942 (Planck15)
    #sm_fwhm = 11.977  # in kpc  -- 1.634" arcsec at z=0.6942 (Planck15)


    xobsbin = spatial_scale_xy[0] * kpc_per_arcsec     # in kpc -- about 0.7" arcsec at z=0.6942
    yobsbin = spatial_scale_xy[1] * kpc_per_arcsec     # in kpc -- about 0.3" arcsec at z=0.6942


    pad_factor = 1.0   # will use pad_factor * nx or pad_factor * ny to pad edges of the cube

    if(infil==None):
        infil = 'spec.fits'

    if(outroot==None):
        outroot = 'spec'

    ### Declare filename directly instead
    #outfil = outroot+'_conv_rebin.fits'

    hdu = fits.open(infil)
    data = hdu[0].data
    wave = hdu[1].data
    #dwv = np.abs(wave[1] - wave[0])
    x_arr = hdu[2].data  # physical distance
    dx = np.abs(x_arr[1] - x_arr[0])  # bin size in kpc

    ny, nx, nlam = data.shape

    # Make padded cube
    pad_x = int(pad_factor * nx)
    pad_y = int(pad_factor * ny)
    data_pad = np.pad(data, ((pad_y,pad_y),(pad_x,pad_x),(0,0)), mode='constant', constant_values=0)
    pny, pnx, pnlam = data_pad.shape
    tot_extent = pnx*dx              # total extent of model plus pad in kpc


    # Remove continuum source
    #r_gal = 0.1      # extent of galaxy in kpc
    #mid_idx = int(pnx/2)
    #gal_sz = np.ceil(r_gal/dx)
    #data_pad[int((mid_idx-gal_sz)):int((mid_idx+gal_sz)),int((mid_idx-gal_sz)):int((mid_idx+gal_sz)),:] = 0.0

    # Define kernel
    pix_fwhm = sm_fwhm / dx
    pix_stddev = pix_fwhm / (2.0 * (2.0*np.log(2.0))**0.5)
    kernel = Gaussian2DKernel(pix_stddev)

    # Do we need to trim before rebinning (probably)
    newshape_x = int(tot_extent / xobsbin)
    newshape_y = int(tot_extent / yobsbin)
    quot_x = pnx / newshape_x
    quot_y = pny / newshape_y
    if pnx % newshape_x != 0:
        xtrimpx = pnx % newshape_x
        xtr1 = int(xtrimpx/2)
        slx = np.s_[xtr1:-(xtrimpx-xtr1)]
    else:
        slx = np.s_[:]
    if pny % newshape_y != 0:
        ytrimpx = pny % newshape_y
        ytr1 = int(ytrimpx/2)
        sly = np.s_[ytr1:-(ytrimpx-ytr1)]
    else:
        sly = np.s_[:]
    data_pad = data_pad[sly,slx,:]
    rb_conv_cube = np.zeros((newshape_y,newshape_x,pnlam))

    # Convolve and rebin
    # New dimensions must divide old ones
    print("Dimension check -- ")
    print("Original array is ", pnx, " by ", pny)
    print("New shape is ", newshape_x, " by ", newshape_y)

    for ilam in range(pnlam):
        conv = convolve_fft(data_pad[:,:,ilam], kernel, normalize_kernel=True)
        rb_conv_cube[:,:,ilam] = utils.bin_ndarray(conv, (newshape_y,newshape_x), operation='sum')
        print("Convolving wavelength ", wave[ilam], "; index = ", ilam)

    print("Transposing cube to (y,x,lam)")
    rb_conv_cube = np.transpose(rb_conv_cube,axes=(1,0,2))

    # Write fits file
    hdulist = None
    arr_list = [data_pad, rb_conv_cube, wave, dx]

    for arr in arr_list:
        if hdulist is None:
            hdulist = fits.HDUList([fits.PrimaryHDU(arr)])
        else:
            hdulist.append(fits.ImageHDU(arr))

    hdulist.writeto(outfil, overwrite=True)




def rtModel_conv_rb_spec(input, outfil=None, specR=1800., zgal=0.6942, \
                         restlam=2796.35, dlam_rb=None,fmt=['lam','y','x'], \
                         return_cube=True):
    '''Convolve and rebin a radiative transfer model cube in the spectral direction

    Parameters
    ----------
    input : str or DataCube
        Input filename or DataCube to convolve/rebin spectrally
    outfil : str, optional
        Output filename
    specR : float, optional
        Resolving power of spectrum
    zgal : float, optional
        Redshift of object
    restlam : float, optional
        Reference rest-frame wavelength for calculating kernel sigma
    dlam_rb : float, optional
        New wavelength bin size; if None, do not rebin
    fmt : list, optional
        Axis order of data cube, must have 'x', 'y', and 'lam' in some order
    return_cube : bool, optional
        If True, return DataCube object


    Returns
    -------
    newwave : array
        Wavelength vector (same as that of input cube if dlam_rb is None)
    specconv_cube : 3D array
        Datacube with convolved spaxels (spatial, spatial, spectral)
    '''
    from linetools.spectra.xspectrum1d import XSpectrum1D

    dlam_obs = restlam * (1.0 + zgal) / specR  # FWHM of resolution element in Angstroms

    if isinstance(input,str):
        ### Assume data format is as the output of
        ### KR's read_spec_output.read_fullfits_conv_rebin()
        hdu = fits.open(input)
        data_pad = hdu[0].data  # Original data, just padded
        cubedat = hdu[1].data  # Spatially convolved data
        wave = hdu[2].data  # Wavelength array
        nx, ny, nlam = cubedat.shape
        cubewcs = None
    else:
        wave = input.wavelength
        cubedat = input.data
        cubewcs = input.wcs

    ### Establish dimensions of input cube
    xs = fmt.index('x')
    ys = fmt.index('y')
    sp = fmt.index('lam')
    cdims = cubedat.shape
    nlam = cdims[sp]
    nx = cdims[xs]
    ny = cdims[ys]

    ### Rearrange the cube (temporarily) : (y,x,lam)
    cubedat_tp = np.transpose(cubedat, axes=[ys, xs, sp])
    fmt_internal = ['y','x','lam']

    ### Set up kernel to convolve spaxels
    dwv = np.abs(wave[1] - wave[0])
    fwhm_specpix = dlam_obs / dwv  # FWHM in number of pixels
    print('FWHM in pixels for convolution:',fwhm_specpix)
    stddev_specpix = fwhm_specpix / (2.0 * (2.0 * np.log(2.0)) ** 0.5)
    spec_kernel = Gaussian1DKernel(stddev_specpix)

    ### Rebin, if desired, and convolve
    if dlam_rb is None:
        specconv_cube = np.zeros((ny, nx, nlam))
        newwave = wave
        for spaxx in range(nx):
            for spaxy in range(ny):
                spax_spec = cubedat_tp[spaxy, spaxx, :]
                #conv_spec = spax_spec # debugging convolution (make res too low?)
                conv_spec = convolve(spax_spec, spec_kernel, normalize_kernel=True)
                specconv_cube[spaxy, spaxx, :] = conv_spec
    else:
        waveq = wave * u.Angstrom
        newwave = np.arange(int(wave[0]),int(wave[-1]),dlam_rb) * u.Angstrom
        specconv_cube = np.zeros((ny, nx, len(newwave)))
        for spaxx in range(nx):
            for spaxy in range(ny):
                spax_spec = cubedat_tp[spaxy, spaxx, :]
                #conv_spec = spax_spec  # debugging convolution (make res too low?)
                conv_spec = convolve(spax_spec, spec_kernel, normalize_kernel=True,
                                     boundary='extend')

                # Use the linetools rebinning method
                csX1d = XSpectrum1D(wave=waveq, flux = conv_spec)
                cs_rb = csX1d.rebin(newwave)
                try:
                    specconv_cube[spaxy, spaxx, :] = cs_rb.flux
                except:
                    import pdb
                    pdb.set_trace()

    if cubewcs is not None:
        cubewcs.wcs.pc[2,2] = dlam_rb

    ### Transpose back to input dimensions
    axs = ['a']*3
    for cc in fmt_internal:
        axs[fmt.index(cc)] = fmt_internal.index(cc)
    newdat = np.transpose(specconv_cube,axes=axs)

    if return_cube:
        newcube = DataCube(wavelength=newwave,data=newdat,
                           include_wcs=cubewcs)
        return newwave,newcube
    else:
        return newwave,specconv_cube


def loadRtModel(biconeangle = 80,incline = 40,extent = 30,modeldir = 'BRP19_conv',
                suffix = 'spec_conv1p6_rebinxy_wcstp_convspecrb.fits', filename=None):
    '''Convolve and rebin a radiative transfer model cube in the spectral direction

    Parameters
    ----------
    biconeangle : int
        Opening angle of the biconical outflow: [10,30,45,60,80]
    incline : int
        Inclination angle of the disk relative to line of sight: [40,60,80]
    extent : int
        Outer extent of outflow (in kpc): [
    Returns
    -------
    modelcube : DataCube
        Radiative transfer model cube
    '''


    if modeldir[-1] != '/':
        modeldir += '/'
    if filename is not None:
        try:
            modelcube=DataCube('{}{}'.format(modeldir,filename))
            return modelcube
        except:
            modelcube=DataCube(filename)
            return modelcube
        else:
            raise ValueError('Model datacube file not found')

    if extent != 30:
        fname = '{}brp19_biconical{}_rw{}_disk_theta{}_{}'.format(modeldir,
                                                                  biconeangle, extent, incline, suffix)
    else:
        fname = '{}brp19_biconical{}_disk_theta{}_{}'.format(modeldir,biconeangle,
                                                             incline,suffix)
    modelcube = DataCube(fname)
    return modelcube


def fitRtModel(model, data, variance, initpars=None, mode='linear',
               **kwargs):
    if initpars is None:
        initpars = (1, 0.0025)

    if mode == 'linear':
        func = err_func_3d
    else:
        raise ValueError('Not prepared for that kind of function')
    sol = optimize.least_squares(func, initpars,
                                 args=(model,data,variance),**kwargs)
    return sol


def err_func_3d(a,model,data,err):
    diff = (data-(a[0]*model+a[1]))/np.sqrt(err)
    res = np.sqrt(np.sum(diff**2,(1,2)))
    #res = diff
    res = diff.flatten()
    return res


def residualDistribution(a, model, data):
    diff = data-(a[0]*model+a[1])
    return diff.flatten()

