import logging
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.time import Time
import astropy.units as u
from datetime import datetime
import ccdproc
from ccdproc import CCDData 
import astrometry # type: ignore
import sep
import numpy as np

# Define scale mappings
# Based on astrometry.net documentation
# https://pypi.org/project/astrometry/#choosing-series
# Maps scale_number -> (min_arcmin, max_arcmin)
SCALE_MAP = {
    0: (2.0, 2.8),
    1: (2.8, 4.0),
    2: (4.0, 5.6),
    3: (5.6, 8.0),
    4: (8.0, 11.0),
    5: (11.0, 16.0),
    6: (16.0, 22.0),
    7: (22.0, 30.0),
    8: (30.0, 42.0),
    9: (42.0, 60.0),
    10: (60.0, 85.0),
    11: (85.0, 120.0),
    12: (120.0, 170.0),
    13: (170.0, 240.0),
    14: (240.0, 340.0),
    15: (340.0, 480.0),
    16: (480.0, 680.0),
    17: (680.0, 1000.0),
    18: (1000.0, 1400.0),
    19: (1400.0, 2000.0),
}

class FitsLv1:
    """
    Class for Level-1 processing of FITS.
    - Find and update WCS information (update_wcs).
    - Bias, dark, and flat corrections (preprocessing).

    It assumes input files are uncompressed FITS.
    """

    def __init__(self, log_file: str = None):
        """
        Initializes the FitsLv1 class and configures logging.

        Parameters
        ----------
        log_file : str, optional
            Path to a file where logs should be saved, in addition to
            streaming to the console.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        
        if log_file:
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(handler.formatter)
            self.logger.addHandler(file_h)

    def _get_dynamic_scales(self, fov_deg: float) -> set:
        """
        Helper function to select index scales based on FOV.
        Uses the 10% to 100% rule, then selects the 3 middle scales.
        """
        
        fov_arcmin = fov_deg * 60.0
        min_search_arcmin = fov_arcmin * 0.1  # 10%
        max_search_arcmin = fov_arcmin * 1.0  # 100%
        
        sorted_scale_items = sorted(SCALE_MAP.items()) # Sort the map by scale number
        
        overlapping_scales = []
        for scale_num, (scale_min, scale_max) in sorted_scale_items:
            if (scale_min <= max_search_arcmin) and (scale_max >= min_search_arcmin):
                overlapping_scales.append(scale_num)
                
        if not overlapping_scales:
            self.logger.warning(f"No matching index scales found for FOV {fov_deg:.3f} deg.")
            return set()

        # Select three middle scales (or all if <=3)
        if len(overlapping_scales) <= 3:
            self.logger.warning(f"Found {len(overlapping_scales)} overlapping scales. Using all: {overlapping_scales}")
            return set(overlapping_scales)
        else:
            mid_index = len(overlapping_scales) // 2
            middle_three_scales = overlapping_scales[mid_index - 1 : mid_index + 2]
            return set(middle_three_scales)

    def update_wcs(self, 
                   fpath_fits, 
                   fpath_out,
                   fov_deg: float = 1.0,
                   pixel_scale_arcsec: float = None,
                   key_ra: str = None, 
                   key_dec: str = None, 
                   search_radius_deg: float = 2.0,
                   cache_directory = "astrometry_cache", 
                   return_fpath: bool = True):
        """
        Solve for WCS and update FITS header.
        Reads .fits and writes to .wcs.fits.
        """
        fpath_fits = Path(fpath_fits)
        fpath_out = Path(fpath_out)
        fpath_out.parent.mkdir(parents=True, exist_ok=True)

        # Read science image
        try:
            sci = CCDData.read(fpath_fits) 
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")
            return

        # Detect sources using SEP
        sci.data = sci.data.astype(np.float32)
        try:
            bkg = sep.Background(sci.data)
            objs = sep.extract(sci.data - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
            bright = objs[np.argsort(objs['flux'])[::-1][:50]]
            coords = np.vstack((bright['x'], bright['y'])).T
        except Exception as e:
            self.logger.error(f"Source detection failed for {fpath_fits.name}: {e}")
            return
        
        # Determine astrometry scales
        dynamic_scales = self._get_dynamic_scales(fov_deg)
        if not dynamic_scales:
            self.logger.error(f"Could not determine scales for FOV {fov_deg} deg. Aborting WCS solve.")
            return

        # Build Index File List (Split between 5200 and 4100 series)
        scales_5200 = {s for s in dynamic_scales if 0 <= s <= 6}  # Series 5200 covers scales 0-6
        scales_4100 = {s for s in dynamic_scales if 7 <= s <= 19} # Series 4100 covers scales 7-19
        
        index_files_list = []
        if scales_5200:
            try:
                files = astrometry.series_5200.index_files(
                    cache_directory=cache_directory, 
                    scales=scales_5200
                )
                index_files_list.extend(files)
            except Exception as e:
                pass

        if scales_4100:
            try:
                files = astrometry.series_4100.index_files(
                    cache_directory=cache_directory, 
                    scales=scales_4100
                )
                index_files_list.extend(files)
            except AttributeError:
                self.logger.warning("astrometry.series_4100 not found. Trying series_4200 (2MASS).")
                try:
                    files = astrometry.series_4200.index_files(
                        cache_directory=cache_directory, 
                        scales=scales_4100
                    )
                    index_files_list.extend(files)
                except Exception as e:
                    pass
            except Exception as e:
                pass

        if not index_files_list:
            self.logger.error("No index files could be selected/downloaded. Aborting.")
            return

        # Dynamically set pixel scale hint (if given)
        if pixel_scale_arcsec:
            # Use a 10% tolerance band around the known pixel scale
            size_hint = astrometry.SizeHint(
                            lower_arcsec_per_pixel=pixel_scale_arcsec * 0.9, 
                            upper_arcsec_per_pixel=pixel_scale_arcsec * 1.1
                        )
        else:
            size_hint = None
        
        # Set position hint (if given)            
        try:
            ra_val = sci.header[key_ra]
            if isinstance(ra_val, str):
                ra_hint = Angle(ra_val, unit=u.hour).degree
            else:
                ra_hint = Angle(ra_val, unit=u.deg).degree

            dec_val = sci.header[key_dec]
            dec_hint = Angle(dec_val, unit=u.deg).degree
            
            position_hint=astrometry.PositionHint(
                    ra_deg=ra_hint,
                    dec_deg=dec_hint,
                    radius_deg=search_radius_deg
                )
        except KeyError as e:
            self.logger.warning(f"Header key {e} not found (for {key_ra}/{key_dec}). Skip position hint.")
            position_hint = None
            
        except (ValueError, TypeError):
            position_hint = None

        # === Solve WCS ===
        try:
            with astrometry.Solver(index_files_list) as solver:
                
                sol = solver.solve(
                    stars=coords,
                    size_hint=size_hint,
                    position_hint=position_hint,
                    solution_parameters=astrometry.SolutionParameters(
                        logodds_callback=lambda l: astrometry.Action.STOP if len(l)>=10 else astrometry.Action.CONTINUE,
                        sip_order=3
                    )
                )
                if sol.has_match():
                    best = sol.best_match()
                    self.logger.info(f"WCS match: RA={best.center_ra_deg:.5f} deg, DEC={best.center_dec_deg:.5f} deg, scale={best.scale_arcsec_per_pixel:.3f} \"/pixel" )
                    sci.wcs = best.astropy_wcs()
                    hdr = best.astropy_wcs().to_header(relax=True)
                    sci.header.extend(hdr, update=True)
                    sci.header['PIXSCALE'] = (best.scale_arcsec_per_pixel, "arcsec/pixel")
                    sci.header['HISTORY'] = f"({datetime.now().isoformat()}) WCS updated. (solopy.Lv1.update_wcs)"
                else:
                    self.logger.warning(f"No WCS solution found for {fpath_fits.name}.")
        except Exception as e:
            self.logger.warning(f"Astrometry.net solver failed for {fpath_fits.name}: {e}")
        # ====================

        # Write updated FITS
        new_hdu = fits.PrimaryHDU(data=sci.data, header=sci.header)
        try:
            new_hdu.writeto(fpath_out, overwrite=True)
            self.logger.info(f"WCS updated: {fpath_out.name}")
        except Exception as e:
            self.logger.error(f"Failed to write {fpath_out}: {e}")
            return

        if return_fpath:
            return fpath_out

    def preprocessing(self, fpath_fits, fpath_out, masterdir, return_fpath=True):
        """
        Subtract bias, dark, mask bad pixels, and flat-correct.
        Reads .fits and writes Multi-Extension .fits.
        Master frames are assumed to be uncompressed .fits files.
        """
        fpath_fits = Path(fpath_fits)
        fpath_out = Path(fpath_out)
        fpath_out.parent.mkdir(parents=True, exist_ok=True)
        masterdir = Path(masterdir)
        
        self.logger.info(f"Starting preprocessing. FILENAME={fpath_fits.name}, MASTERDIR={masterdir}.")
        
        try:
            sci = CCDData.read(fpath_fits)
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading science frame: {e}")
            return
        
        try:
            mbias, fname_mbias = self._select_master(masterdir, 'BIAS', sci.header['JD'], return_fname=True)
            mdark, fname_mdark = self._select_master(masterdir, 'DARK', sci.header['JD'], sci.header['EXPTIME'], return_fname=True)
            mflat, fname_mflat = self._select_master(masterdir, 'FLAT', sci.header['JD'], return_fname=True)
        except Exception as e:
            self.logger.error(f"Failed to load master frames: {e}")
            return
        
        try:
            mask, fname_mask = self._select_master(masterdir, 'MASK', sci.header['JD'], return_fname=True)
        except Exception as e:
            self.logger.warning(f"Failed to load master mask. Proceeding without mask.")
            mask, fname_mask = None, None
        
        try:
            # Bias
            bsci = ccdproc.subtract_bias(sci, mbias)
            bsci.meta['BIASCORR'] = (True, "Bias corrected?")
            bsci.meta['BIASNAME'] = (fname_mbias, "File name of master bias")
            bsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bias subtracted. ({fname_mbias})"
            self.logger.info(f"Bias subtracted.")
            
            # Dark
            bdsci = ccdproc.subtract_dark(bsci, mdark, exposure_time="EXPTIME", exposure_unit=u.second)
            bdsci.meta['DARKCORR'] = (True, "Dark corrected?")
            bdsci.meta['DARKNAME'] = (fname_mdark, "File name of master dark")
            bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Dark subtracted. ({fname_mdark})"
            self.logger.info(f"Dark subtracted.")
            
            # Mask (Mask negative values and clip to 0)
            mask_negative = bdsci.data < 0
            if mask is not None:
                mask = mask_negative | mask.data
                bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bad pixels masked. ({fname_mask})"
                bdsci.meta['MASKNAME'] = (fname_mask, "File name of bad pixel mask")
            else:
                mask = mask_negative
            bdsci.mask = mask
            bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Negative values masked."
            self.logger.info(f"Bad pixels masked.")
            
            # Flat
            psci = ccdproc.flat_correct(bdsci, mflat)
            psci.meta['FLATCORR'] = (True, "Flat corrected?")
            psci.meta['FLATNAME'] = (fname_mflat, "File name of master flat")
            psci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Flat corrected. ({fname_mflat})"
            self.logger.info(f"Flat corrected.")
            
            # Metadata updates
            if 'FILENAME' in psci.meta:
                psci.meta['HISTORY'] = f"File name updated. ({psci.meta['FILENAME']} -> {fpath_out.stem})"
            psci.meta['FILENAME'] = fpath_out.stem
            
        except Exception as e:
            self.logger.error(f"CCD processing failed: {e}")
            return
        
        # Write Multi-Extension FITS
        # 1. Primary HDU
        primary_hdu = fits.PrimaryHDU(data=psci.data, header=psci.header)
        
        # 2. Mask HDU (if available)
        if psci.mask is not None:
            mask_data = psci.mask.astype(np.uint8)
            mask_hdu = fits.ImageHDU(data=mask_data, name='MASK')
            hdul_out = fits.HDUList([primary_hdu, mask_hdu])
        else:
            hdul_out = fits.HDUList([primary_hdu])
        
        try:
            hdul_out.writeto(fpath_out, overwrite=True)
            self.logger.info(f"Preprocessing complete: {fpath_out.name}")
        except Exception as e:
            self.logger.error(f"Failed to write FITS ({fpath_out.name}): {e}")
            return
        
        if return_fpath:
            return fpath_out

    def _select_master(self, masterdir, imagetyp, jd_target, exptime=None, return_fname=False):
        """
        Helper to select the closest master frame by JD (and exposure time if given).
        Assumes master frames are UNCOMPRESSED .fits files.
        """
        coll = ccdproc.ImageFileCollection(masterdir, glob_include="*.fits").filter(imagetyp=imagetyp)
        if not coll.files:
            raise FileNotFoundError(f"No master {imagetyp} frames found in {masterdir}")

        df = coll.summary.to_pandas()
        
        # Find 'jd' or 'JD' column
        jd_col = None
        if 'jd' in df.columns:
            jd_col = 'jd'
        elif 'JD' in df.columns:
            jd_col = 'JD'
        else:
            raise KeyError(f"Cannot find 'JD' columns for master {imagetyp} frames.")

        df[jd_col] = df[jd_col].astype(float)
        df['diff'] = (df[jd_col] - jd_target).abs()
        
        if exptime is not None:
            # Find exptime column
            exp_col = None
            if 'exptime' in df.columns:
                exp_col = 'exptime'
            elif 'EXPTIME' in df.columns:
                exp_col = 'EXPTIME'
            
            if exp_col:
                df[exp_col] = df[exp_col].astype(float)
                # Select the master dark frame with the closest exposure time
                exptime_unique = df[exp_col].unique()
                exptime_closest = min(exptime_unique, key=lambda x: abs(x - float(exptime)))
                self.logger.info(f"Requested exptime={exptime:.1f} sec, found closest master frame exptime={exptime_closest:.1f} sec.")
                df = df[df[exp_col] == exptime_closest].copy() # .copy() to avoid SettingWithCopyWarning
            else:
                self.logger.warning(f"Cannot find 'exptime' columns for master {imagetyp}, proceeding without exptime filter.")

        if df.empty:
            raise FileNotFoundError(f"No matching master {imagetyp} frame found for JD={jd_target:.1f}, EXPTIME={exptime:.1f} sec.")

        idx = df['diff'].idxmin()
        selected_file = Path(coll.files_filtered(include_path=True)[idx])
        self.logger.info(f"Selected master {imagetyp}: {selected_file.name}")
            
        try:  
            if imagetyp == 'MASK':   
                master_frame = CCDData.read(selected_file, unit='bool')
            else:
                master_frame = CCDData.read(selected_file)
        except ValueError:
            master_frame = ccdproc.CCDData.read(selected_file, unit='adu')

        if return_fname:
            return master_frame, selected_file.name
        else:
            return master_frame