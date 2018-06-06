import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve,convolve_fft
from scipy import optimize
from reproject import reproject_interp
import pdb

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
        initpars = np.array([init_norm,init_fwhm[0],init_fwhm[1],
                             init_theta,init_offset])
    else:
        ef = err_func
        initpars = np.array([init_norm, init_fwhm,init_offset])

    sol = optimize.least_squares(ef,initpars,bounds=parbounds,
            args =(highres,lowres,wcs_hires,wcs_lowres,pixscale_highres,
                   np.array(fitbounds)),**kwargs)
    return sol


def err_func(pars,inp,data,wcs_hires,wcs_lowres,pixscale=0.03,
             bounds=np.array([[6,20],[42,57]])):
    norm,fwhm,offset = pars
    numseepix = fwhm / pixscale
    nanpix_data = np.where(np.isnan(data))
    data[nanpix_data]=0

    kern = Gaussian2DKernel(numseepix / 2.355)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    convdat = norm*convolve_fft(inp, kern) + offset
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
    norm,fwhmx,fwhmy,theta, offset = pars
    numseepix_x = fwhmx / pixscale
    numseepix_y = fwhmy / pixscale
    nanpix_data = np.where(np.isnan(data))
    data[nanpix_data]=0

    kern = Gaussian2DKernel(numseepix_x / 2.355,numseepix_y / 2.355,
                            theta=theta)  # FWHM = sigma * 2 sqrt(2.*ln(2))
    #convdat = norm*convolve_fft(inp, kern,boundary='wrap')
    convdat = norm * convolve_fft(inp, kern, boundary='wrap') +offset

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
        fwhmx, fwhmy = fwhm
        numseepix_x = fwhmx / pixscale
        numseepix_y = fwhmy / pixscale
        if angle is None:
            angle = 0
        kern = Gaussian2DKernel(numseepix_x / 2.355, numseepix_y / 2.355,theta=angle)
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
    inpreproj, inpfootprint = \
        reproject_interp((convdat, wcs_hires), wcs_lowres, order='nearest-neighbor',
                         shape_out=np.shape(data))
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

