import numpy as np
import numpy.polynomial.legendre as L
from kcwi_jnb.utils import closest
from scipy import stats

def initcont(wave, flux, sigup, wave1, wave2, kernsize=10, maxorder=None):
    waveidx1 = closest(wave, wave1)
    waveidx2 = closest(wave, wave2)
    wrange = range(waveidx1, waveidx2)
    ### Generate automatic first-pass fit
    # Initialize points for fit
    initcontpts = np.min((len(wrange), 400))
    ptspacing = (wave2 - wave1) / initcontpts
    initx = wave1 + np.arange(initcontpts) * ptspacing
    initidx = []
    for i in range(initcontpts):
        initidx.append(closest(wave, initx[i]))
    convtrim = int(np.ceil(kernsize / 2.))
    initidx = initidx[convtrim:-convtrim]
    fitidx = initidx
    initwave = wave[initidx]
    convflux = np.convolve(flux, np.ones(kernsize) / kernsize, mode='same')
    initflux = convflux[initidx]
    initsig = sigup[initidx]

    # If global max order imposed, set up values for each iteraton
    if maxorder is None:
        mo = [4, 5, 5, 6]
    else:
        mo = [maxorder, maxorder, maxorder, maxorder]

    # Fit initial points and reject fit points far from continuum
    coeff, covmtx = fitcont(wave, flux, sigup, fitidx, mo[0])
    cont = L.legval(initwave, coeff)
    resarr = np.abs(cont - initflux)
    toremove = []
    for i in range(len(fitidx)):
        if resarr[i] > (1. * initsig[i]):
            toremove.append(i)
    fitidx = np.delete(fitidx, toremove)
    contwave = wave[fitidx]
    contflux = flux[fitidx]
    contsig = sigup[fitidx]

    # Fit again and reject again
    coeff, covmtx = fitcont(wave, flux, sigup, fitidx, mo[1])
    cont = L.legval(contwave, coeff)
    resarr = np.abs(cont - contflux)
    toremove = []
    for i in range(len(cont)):
        if resarr[i] > (1. * contsig[i]):
            toremove.append(i)
    fitidx = np.delete(fitidx, toremove)

    # Now fit again and using all initial x-values as domain for continuum, reject intial pts from new cont.
    coeff, covmtx = fitcont(wave, flux, sigup, fitidx, mo[2])
    cont = L.legval(initwave, coeff)
    resarr = np.abs(cont - initflux)
    toremove = []
    for i in range(len(initsig)):
        if resarr[i] > (1. * initsig[i]):
            toremove.append(i)
    fitidx = np.delete(initidx, toremove)

    # Final first-pass fit!
    coeff, covmtx = fitcont(wave, flux, sigup, fitidx, mo[3])
    return coeff, covmtx


def fitcont(wave, flux, sigup, fitidx, maxorder=6):
    wavepts = wave[fitidx]
    fluxpts = flux[fitidx]
    sigpts = sigup[fitidx]
    ### Cycle through fits of several orders and decide the appropriate order by F-test
    coeffs = []
    covmtxs = []
    fits = []
    chisqs = []
    dfs = []
    fprob = 0.
    i = 0
    order = 1
    while (fprob <= 0.95) & (order <= maxorder):
        order = i + 1
        coeff, covmtx = fitlegendre(wavepts, fluxpts, sigpts, order)
        coeffs.append(coeff)
        covmtxs.append(covmtx)
        fit = L.legval(wavepts, coeff, order)
        fits.append(fit)
        chisq, df = redchisq(fluxpts, fit, sigpts, order)
        chisqs.append(chisq), dfs.append(df)
        if i > 0:
            fval = chisqs[i] / chisqs[i - 1]
            fprob = stats.f.cdf(fval, dfs[i], dfs[i - 1])
        i += 1
    ### Choose fit of order just prior to where F-test indicates no improvement
    fitcoeff = coeffs[i - 2]
    fitcovmtx = covmtxs[i - 2]
    wrange = range(fitidx[0], fitidx[-1])
    cont = L.legval(wave[wrange], fitcoeff)
    return fitcoeff, fitcovmtx


def evalcont(wave, coeff):
    cont = L.legval(wave, coeff)
    return cont


def fitlegendre(wavepts, fluxpts, sigpts, order):
    from numpy.linalg import svd
    vander = L.legvander(wavepts, order)
    design = np.zeros(vander.shape)
    for i in range(len(wavepts)):
        design[i] = vander[i] / sigpts[i]
    U, s, v = svd(design, full_matrices=False, compute_uv=True)
    V = v.transpose()
    solvec = np.zeros(order + 1)
    for i in range(order + 1):
        solvec += np.dot(U[:, i], fluxpts / sigpts) / s[i] * V[:, i]
    ### Build covariance matrix
    covmtx = np.zeros([order + 1, order + 1])
    for j in range(order + 1):
        for k in range(order + 1):
            covmtx[j, k] = sum(V[:, j] * V[:, k] / s)
    return solvec, covmtx


def redchisq(fluxpts1, fitpts1, sigpts, order):
    diff = sum(fluxpts1 - fitpts1)
    df = len(fluxpts1) - order - 1
    Xsq = 1. / df * sum(diff ** 2 / sigpts ** 2)
    return Xsq, df

### Transform wavelength into velocity space centered on some line
def veltrans(redshift,waves,line):
    c=299792.458
    if (isinstance(line,int)|isinstance(line,float)):
        transline =c*(waves-(1+redshift)*line)/line/(1+redshift)
    else:
        transline=[]
        for ll in line: transline.append(c*(waves-(1+redshift)*ll)/ll/(1+redshift))
    return transline
