import numpy as np
from kcwi_jnb import utils as ku
from astropy.wcs import WCS

class DataCube(object):
    def __init__(self,inp):
        if isinstance(inp,str):
            dat,wcs = ku.load_file(inp)
            filename = inp
            hdr = wcs.to_header()
        else:
            dat = inp[0].data
            hdr = inp[0].header
            wcs = WCS(hdr)
            filename = ''

        self.data = dat
        self.wcs = wcs
        self.header = hdr
        self.wavelength = ku.get_wave_arr(dat,wcs,extract_wcs=False)
        self.filename = filename

    def write(self,filename):
        newhdu = self.wcs.to_fits()
        newhdu[0].data = self.data
        newhdu.writeto(filename,overwrite=True)

    def copy(self):
        return