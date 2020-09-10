import numpy as np
from kcwi_jnb import utils as ku
from astropy.wcs import WCS

class DataCube(object):
    def __init__(self,inp=None,data=None,wavelength=None,include_wcs=True):
        if inp is None:
            dat = data
            filename = ''
            hdr = None
        elif isinstance(inp,str):
            dat,wcs = ku.load_file(inp)
            filename = inp
            hdr = wcs.to_header()
            #self.wcs = wcs
        else:
            dat = inp[0].data
            hdr = inp[0].header
            wcs = WCS(hdr)
            filename = ''

        self.data = dat
        self.header = hdr
        if include_wcs is False:
            self.wcs = None
            self.wavelength = wavelength
        elif include_wcs is True:
            self.wcs = wcs
            if len(np.shape(dat)) == 3:
                self.wavelength = ku.get_wave_arr(dat,self.wcs,extract_wcs=False)
            else:
                self.wavelength = None
        else:
            self.wcs = include_wcs
            self.header = self.wcs.to_header()
            if wavelength is None:
                self.wavelength = ku.get_wave_arr(dat, self.wcs, extract_wcs=False)
            else:
                self.wavelength=wavelength

        self.filename = filename

    def write(self,filename,instrument='KCWI'):
        newhdu = self.wcs.to_fits()
        newhdu[0].data = self.data
        newhdu[0].header['INSTRUME'] = 'KCWI'
        try:
            newhdu.writeto(filename,overwrite=True)
        except:
            import pdb; pdb.set_trace()

    def copy(self):
        newcube = DataCube(data=self.data.copy(), wavelength=self.wavelength.copy(), include_wcs=self.wcs.copy())
        newcube.header = self.header.copy()
        return newcube