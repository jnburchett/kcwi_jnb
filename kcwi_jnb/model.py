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

def addWcsRtModel(inp,ref,outfile,spatial_scale=0.2,zgal=0.6942,
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

    ### Rearrange the cube
    newdat= np.transpose(rtcube.data, axes=[2, 1, 0])


    ### Get the pixel scale in the spectral dimension
    diff = wavearr[1:]-wavearr[:-1]
    if not np.all(diff == diff[0]):
        raise ValueError('Wavelength array not uniformly sampled')
    spectral_scale = diff[0]

    ### Set scale and reference positions
    wave1 = wavearr[0]*(1.+zgal)
    spectral_scale *= (1.+zgal)

    ### Get the bright pixel from reference cube
    brightpix, brightcoords = utils.bright_pix_coords(refnb, refwcs)

    ### Create the new WCS
    newwcs = WCS(naxis=3)
    spatscaledeg = spatial_scale * cosmo.arcsec_per_kpc_proper(zgal) / 3600.
    print(spatscaledeg)
    if center is not None:
        cencoords = SkyCoord(center[0],center[1],unit=(u.hourangle,u.deg))
    else:
        cencoords = brightcoords
    newwcs.wcs.crpix = np.array([150.5, 150.5, 1])
    newwcs.wcs.crval = np.array([cencoords.ra.deg, cencoords.dec.deg, wave1])
    newwcs.wcs.cdelt = np.array([8.0226e-6, 8.0226e-6, spectral_scale])
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


def rtModel_conv_rb_spec(input, outfil=None, specR=1800., zgal=0.6942,
                      restlam=2796.35, dlam_rb=None,fmt=['lam','y','x'],
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

    dlam_obs = restlam * (1.0 + zgal) / specR
    dlam_rest = dlam_obs / (1.0 + zgal)  # FWHM of resolution element in Angstroms

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
        wave = input.wavelength / (1. + zgal)
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
    fwhm_specpix = dlam_rest / dwv  # FWHM in number of pixels
    stddev_specpix = fwhm_specpix / (2.0 * (2.0 * np.log(2.0)) ** 0.5)
    spec_kernel = Gaussian1DKernel(stddev_specpix)

    ### Rebin, if desired, and convolve
    if dlam_rb is None:
        specconv_cube = np.zeros((ny, nx, nlam))
        newwave = wave
        for spaxx in range(nx):
            for spaxy in range(ny):
                spax_spec = cubedat_tp[spaxy, spaxx, :]
                conv_spec = convolve(spax_spec, spec_kernel, normalize_kernel=True)
                specconv_cube[spaxy, spaxx, :] = conv_spec
    else:
        waveq = wave * u.Angstrom
        newwave = np.arange(int(wave[0]),int(wave[-1]),dlam_rb) * u.Angstrom
        specconv_cube = np.zeros((ny, nx, len(newwave)))
        for spaxx in range(nx):
            for spaxy in range(ny):
                spax_spec = cubedat_tp[spaxy, spaxx, :]
                conv_spec = convolve(spax_spec, spec_kernel, normalize_kernel=True)
                # Use the linetools rebinning method
                csX1d = XSpectrum1D(wave=waveq, flux = conv_spec)
                cs_rb = csX1d.rebin(newwave)
                try:
                    specconv_cube[spaxy, spaxx, :] = cs_rb.flux
                except:
                    import pdb
                    pdb.set_trace()

    if cubewcs is not None:
        cubewcs.wcs.pc[2,2] = dlam_rb  ### Probably shouldn't be hard-coded

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



def fitRtModel():
    return

