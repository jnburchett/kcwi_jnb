import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from kcwi_jnb import transform as kt
from kcwi_jnb.cube import DataCube
from kcwi_jnb import spectral
from kcwi_jnb.utils import veltrans,closest
from astropy.cosmology import Planck15 as cosmo

from linetools.lists.linelist import LineList

def continuum_subtract(cubefile, varcube, wave1, wave2, outfile=None,
                       normalize=False, return_sig=True, flat_cont=None,
                       return_cont = False):
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
    hdr = slicedhdu.header
    wcs = WCS(hdr)
    dat = slicedhdu.data
    var = slicedvar.data
    datshape = np.shape(dat)
    try:
        wdelt = hdr['PC3_3']
    except:
        wdelt = hdr['CDELT3']
    if flat_cont is None:
        dim = np.shape(dat)
        lastwaveidx = dim[0] - 1
        w1 = wcs.wcs_pix2world([[0, 0, 0]], 1)[0][2]
        w2 = wcs.wcs_pix2world([[0, 0, lastwaveidx]], 1)[0][2]
    else:
        w1 = flat_cont[0]
        w2 = flat_cont[1]
    wavearr = np.arange(w1, w2 + wdelt, wdelt)
    idxgrid = np.meshgrid(range(datshape[1]), range(datshape[2]))
    idxgrid = np.array(idxgrid)
    idxs2iter = np.reshape(idxgrid.T, (datshape[1] * datshape[2], 1, 2))
    medconts = []

    contcube = np.zeros_like(dat.data)
    for i, idi in enumerate(idxs2iter):
        idx0 = idi[0][0]
        idx1 = idi[0][1]
        thisspec = dat[:, idx0, idx1]
        thisvar = np.sqrt(var[:, idx0, idx1])
        if np.all(np.isnan(thisspec)):
            dat[:, idx0, idx1] = 0
        else:
            try:
                if len(wavearr)>len(thisspec):
                    wavearr=wavearr[:-1]
                contcoeff, contcovmtx = spectral.initcont(wavearr, thisspec, thisvar, w1, w2)
                cont = spectral.evalcont(wavearr, contcoeff)

                if flat_cont is not None:
                    cont = np.median(cont)
                medconts.append(np.median(cont))

                if normalize:
                    dat[:, idx0, idx1] = thisspec / cont
                    var[:, idx0, idx1] = thisvar / cont
                else:
                    dat[:, idx0, idx1] = thisspec - cont
                    var[:, idx0, idx1] = thisvar
            except:
                dat[:, idx0, idx1] = 0
            contcube[:, idx0, idx1] = cont
    slicedhdu.data = dat
    slicedvar.data = var
    if outfile is not None:
        slicedhdu.write(outfile)
    if return_sig&return_cont:
        return slicedhdu,slicedvar,contcube
    elif return_sig:
        return slicedhdu,slicedvar
    elif return_cont:
        return slicedhdu,contcube
    else:
        return slicedhdu



def equiv_width_map(cube, varcube,  measure_range, contwave1=None, contwave2=None,
                    transition='MgII 2796', zsys=0.6942,normalize_continuum=True):
    """Calculate the equivalent width over some wavelength range
    """
    from linetools.spectralline import AbsLine
    if normalize_continuum:
        normcube,signormcube = continuum_subtract(cube,varcube,contwave1,contwave2,
                                  normalize=True)
    else:
        normcube = cube
        signormcube = varcube

    al = AbsLine(transition,z = zsys)
    try:
        if measure_range[0].unit == u.AA:
            #import pdb; pdb.set_trace()
            al.limits.set(((measure_range[0]/al.wrest-1.).value,(measure_range[1]/al.wrest-1.).value))
        elif measure_range.unit == (u.km/u.s):
            al.limits.set(measure_range)
    except:
        raise ValueError("measure_range should be a quantity with velocity or wavelength units")

    cubedims = np.shape(cube.data)
    ys = np.arange(cubedims[2])
    xs = np.arange(cubedims[1])
    grdTarr = np.array(np.meshgrid(xs, ys)).T
    rows = np.shape(grdTarr)[0] * np.shape(grdTarr)[1]
    spcoords = grdTarr.reshape(rows, 2)

    ewarr = np.zeros((cubedims[1],cubedims[2]))
    sigewarr = np.zeros((cubedims[1],cubedims[2]))

    for i,cc in enumerate(spcoords):

        spec = extract_spectrum(cube,cc)
        sigspec = extract_spectrum(varcube,cc)
        spec.sig = sigspec.flux

        al.analy['spec'] = spec
        al.measure_restew()
        ewarr[cc[0],cc[1]] = al.attrib['EW'].value
        sigewarr[cc[0],cc[1]] = al.attrib['sig_EW'].value

    return ewarr, sigewarr

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




def extract_spectrum(cube,pixels,wvslice=None,method='median',varcube=None, spatial_scale_xy=None):

    ## KHRR: if spatial_scale_xy is not None, flux in units of cgs / sq. arcsec are assumed
    ## This is now only implemented with the 'sum' method

    from linetools.spectra.xspectrum1d import XSpectrum1D

    if isinstance(cube,str):
        cube = DataCube(cube)

    if wvslice is not None:
        cube = kt.slice_cube(cube,wvslice[0],wvslice[1])
        if varcube is not None:
            varcube = kt.slice_cube(varcube, wvslice[0], wvslice[1])

    ##import pdb;
    #pdb.set_trace()

    dat = cube.data

    # Get rid of NaNs
    dc = dat.copy()
    nans = np.where(np.isnan(dc))
    dc[nans]=0

    # Perform the extraction
    if method == 'median':
        if np.shape(np.array(pixels))==(2,):
            extspec = dc[:,pixels[0],pixels[1]]
        else:
            spaxels = []
            for i,px in enumerate(pixels):
                thisspec = dc[:,px[0],px[1]]
                spaxels.append(thisspec)
            spaxels=np.array(spaxels)
            extspec = np.median(spaxels, axis=0)
            sigspec = -99. * np.ones(np.shape(dc)[0])
    elif method == 'mean':
        if np.shape(np.array(pixels))==(2,):
            extspec = dc[:,pixels[0],pixels[1]]
        else:
            spaxels = []
            for i,px in enumerate(pixels):
                thisspec = dc[:,px[0],px[1]]
                spaxels.append(thisspec)
            spaxels=np.array(spaxels)
            extspec = np.mean(spaxels, axis=0)
            sigspec = -99. * np.ones(np.shape(dc)[0])
    elif method == 'sum':
        if np.shape(np.array(pixels))==(2,):
            extspec = dc[:,pixels[0],pixels[1]]
        else:
            spaxels = []
            for i,px in enumerate(pixels):
                thisspec = dc[:,px[0],px[1]]
                spaxels.append(thisspec)
            spaxels=np.array(spaxels)
            sigspec = -99. * np.ones(np.shape(dc)[0])

            if spatial_scale_xy is None:
                extspec = np.sum(spaxels, axis=0)
            else:
                spxsz = spatial_scale_xy[0] * spatial_scale_xy[1]
                extspec = spxsz * np.sum(spaxels, axis=0)

    else:  # do the weighted mean
        vardat = varcube.data
        vdc = vardat.copy()
        if np.shape(np.array(pixels))==(2,):
            extspec = dc[:,pixels[0],pixels[1]]
        else:
            spaxels = []
            varspaxels = []
            for i, px in enumerate(pixels):
                thisspec = dc[:, px[0], px[1]]
                thisvar = vdc[:, px[0], px[1]]
                spaxels.append(thisspec)
                varspaxels.append(thisvar)
            spaxels = np.array(spaxels)
            ivarspaxels = 1./np.array(varspaxels)
            extspec = np.sum(spaxels*ivarspaxels, axis=0)/np.sum(ivarspaxels,axis=0)
            sigspec = 1./np.sqrt(np.sum(ivarspaxels,axis=0))
    # Get the wavelength array
    newwavearr = cube.wavelength

    # Create the spectrum
    try:
        spec = XSpectrum1D(wave = newwavearr,flux=extspec,sig=sigspec)
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

def radial_profile_angdist(data, var=None, radii=0.7, center=None):
    if center is None:
        center = np.where(data==np.max(data))
    ypix = np.arange(0,np.shape(data)[0])
    xpix = np.arange(0,np.shape(data)[1])
    dists = np.zeros(np.shape(data))
    for i,xx in enumerate(xpix):
        for j,yy in enumerate(ypix):
            xdist = np.abs(center[1]-xx)*0.7
            ydist = np.abs(center[0]-yy)*0.3
            dists[j,i] = np.sqrt(xdist**2+ydist**2)
    distarr = np.arange(0,np.max(xdist)+radii/2.,radii)
    fluxprofile = np.zeros(len(distarr))
    surfbright = np.zeros(len(distarr))
    varsum = np.zeros(len(distarr))
    varsurfbright = np.zeros(len(distarr))
    for i,dd in enumerate(distarr):
        if i==0:
            thesepix = np.where(dists<dd)
        else:
            thesepix = np.where((dists<dd)&(dists>distarr[i-1]))
        fluxprofile[i] = np.sum(data[thesepix])
        surfbright[i] = fluxprofile[i]/(len(thesepix[0])*0.21) #0.21 arcsec^2 is 1 pixel's area
        if var is not None:
            varsum[i] = np.sum(var[thesepix])
            varsurfbright[i] = varsum[i]/(len(thesepix[0])*0.21)
    if var is not None:
        return distarr,fluxprofile,surfbright,varsum,varsurfbright
    else:
        return distarr,fluxprofile,surfbright

def position_velocity(cube,center,regionwidth=3.,regionextent=20.*u.kpc,
                      velocitylimit=1000.,redshift = 0.6942,restwave=2796.3543):
    velarr = spectral.veltrans(redshift,cube.wavelength,restwave)
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
    veloffset = spectral.veltrans(redshift,specdata[0].wavelength.value,restwave)
    pvarr = np.array([sd.flux for sd in specdata])
    return kpcoffset,veloffset,pvarr

def signif_cube(fluxcube, varcube,wave1,wave2,outfile=None):
    nbhdu, errhdu = kt.narrowband(fluxcube,wave1,wave2,varcube=varcube)
    signifdat = nbhdu.data/np.sqrt(errhdu.data)
    newhdu = nbhdu.copy()
    newhdu.data = signifdat
    if outfile is not None:
        fits.writeto(data=newhdu.data, header=newhdu.header,
                     filename=outfile, overwrite=True)
        #newhdu.writeto(outfile,overwrite=True)
    return newhdu



