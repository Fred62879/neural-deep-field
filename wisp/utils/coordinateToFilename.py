import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pickle
import glob
import os
import time
from astropy.wcs.utils import skycoord_to_pixel
import numpy as np


def coordinateToFilename(ra, dec, size=64,
        fits_file_directories=['/arc/projects/ots/pdr3_dud/calexp-HSC*.fits'],
        crval_cache_file='crval_catalogue.pkl',
        rough_frame_size=0.1*u.deg,
        centre_pixel=2000
    ):
    """
    Note: if a coordinate falls to multiple frames, may return multiple cutouts.
    """
    start = time.time()
    coordinate = SkyCoord(ra*u.deg, dec*u.deg)
    # Build catalogue of RA/DEC positions of all files
    if not os.path.exists(crval_cache_file):
        files = []
        for path in fits_file_directories:
            files.extend(glob.glob(path))

        files = np.array(sorted(files))

        ras = np.zeros(len(files), dtype='f')
        decs = np.zeros(len(files), dtype='f')

        print(f"Building ra/dec catalogue, iterating over fits headers...")
        for i, f in enumerate(files):
            with fits.open(f) as hdul:
                #ras[i] = float(hdul[1].header['CRVAL1'])
                #decs[i] = float(hdul[1].header['CRVAL2'])
                w = WCS(hdul[1].header)
                sky = w.pixel_to_world(centre_pixel, centre_pixel)
                ras[i] = float(sky.ra.deg)
                decs[i] = float(sky.dec.deg)
            if i % 100 == 0:
                print(f"Finished {i}...")
        with open(crval_cache_file, 'wb') as f:
            pickle.dump((ras, decs, files), f)
        print(f"Done.")
    else:
        # print(f"Loaded crval1, crval2, and file list from {crval_cache_file}")
        #start = time.time()
        with open(crval_cache_file, 'rb') as f:
            ras, decs, files = pickle.load(f)
        #end = time.time()
        #print(f'{end-start} seconds')

    end = time.time()
    #print(f"{end-start} setup")
    start = time.time()
    # Now build the catalogue and find closest members
    catalog = SkyCoord(ra=ras*u.degree, dec=decs*u.degree)
    #print(coordinate.separation(catalog) < 0.01*u.deg)
    #print(catalog[coordinate.separation(catalog) < 0.01*u.deg])
    close_files = files[coordinate.separation(catalog) < rough_frame_size]
    #print(len(close_files))
    end = time.time()
    #print(f"{end-start} close files")

    # Do a final check to make sure in footprint
    start = time.time()
    
    # Output is of form ((ra, dec), (file, x, y))
    output = []
    
    for f in close_files:
        with fits.open(f) as hdul:
            w = WCS(hdul[1].header)
            if w.footprint_contains(coordinate):
                x, y = skycoord_to_pixel(coordinate, w)
                output.append(((ra, dec,), (f, int(x), int(y))))
    end = time.time()
    #print(f"{end-start} final check")

    return output

if __name__ == "__main__":
    with fits.open('/arc/projects/ots/pdr3_dud/calexp-HSC-I-9707-4%2C0.fits') as hdul:
        w = WCS(hdul[1].header)
        sky = w.pixel_to_world(2000, 2000)
        print(sky.ra.deg)
        print(sky.dec.deg)
        ra = sky.ra.deg
        dec = sky.dec.deg
        
    print(f"((ra,dec), (filename, x, y))")
    for f in coordinateToFilename(ra,dec):
        print(f)
