import os, sys, argparse, collections
import h5py, hdf5plugin
import numpy as np
import multiprocessing as mp

def init_argparser():
    parser = argparse.ArgumentParser(description = 'Convert SPring-8/BL02B1 Eiger2 CdTe 1M .h5 data to Bruker .sfrm')
    parser.add_argument('-f', required=True,  default='',       metavar='path',  type=str,   dest='_H5F', help='h5 master file [required]')
    parser.add_argument('-o', required=False, default='.',      metavar='path',  type=str,   dest='_OUT', help='sfrm path [\'.\']')
    parser.add_argument('-n', required=False, default='test',   metavar='name',  type=str,   dest='_NAM', help='sfrm name [test]')
    parser.add_argument('-r', required=False, default=1,        metavar='int',   type=int,   dest='_RUN', help='Run number [1]')
    parser.add_argument('-s', required=False, default=1,        metavar='int',   type=int,   dest='_SUM', help='Number of images to sum [1]')
    parser.add_argument('-a', required=False, default=0.001,    metavar='float', type=float, dest='_OSR', help='Omega scan range deg/image [0.001]')
    parser.add_argument('-c', required=False, default=0.0,      metavar='float', type=float, dest='_CHI', help='Chi angle [0.0]')
    parser.add_argument('-t', required=False, default=0.0,      metavar='float', type=float, dest='_TTH', help='2-Theta angle [0.0]')
    parser.add_argument('-w', required=False, default=0.245479, metavar='float', type=float, dest='_WAV', help='Wavelength [0.245479] Ang.')
    # print help if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        raise SystemExit
    return parser.parse_args()

def convert_SP8_Eiger2_Bruker(fnam, inum, isum, iexp, iswp, idat, src_wav=0.245479, gon_chi=0.0, gon_phi=0.0, gon_tth=0.0, gon_dxt=103.5124, det_max=65535, x=508.6675, y=469.8171):

    def bruker_header():
        header = collections.OrderedDict()
        header['FORMAT']  = np.array([100], dtype=np.int64)                       # Frame Format -- 86=SAXI, 100=Bruker
        header['VERSION'] = np.array([18], dtype=np.int64)                        # Header version number
        header['HDRBLKS'] = np.array([15], dtype=np.int64)                        # Header size in 512-byte blocks
        header['TYPE']    = ['Some Frame']                                        # String indicating kind of data in the frame
        header['SITE']    = ['Some Site']                                         # Site name
        header['MODEL']   = ['?']                                                 # Diffractometer model
        header['USER']    = ['USER']                                              # Username
        header['SAMPLE']  = ['']                                                  # Sample ID
        header['SETNAME'] = ['']                                                  # Basic data set name
        header['RUN']     = np.array([1], dtype=np.int64)                         # Run number within the data set
        header['SAMPNUM'] = np.array([1], dtype=np.int64)                         # Specimen number within the data set
        header['TITLE']   = ['', '', '', '', '', '', '', '', '']                  # User comments (8 lines)
        header['NCOUNTS'] = np.array([-9999, 0], dtype=np.int64)                  # Total frame counts, Reference detector counts
        header['NOVERFL'] = np.array([-1, 0, 0], dtype=np.int64)                  # SAXI Format: Number of overflows
                                                                                # Bruker Format: #Underflows; #16-bit overfl; #32-bit overfl
        header['MINIMUM'] = np.array([-9999], dtype=np.int64)                     # Minimum pixel value
        header['MAXIMUM'] = np.array([-9999], dtype=np.int64)                     # Maximum pixel value
        header['NONTIME'] = np.array([-2], dtype=np.int64)                        # Number of on-time events
        header['NLATE']   = np.array([0], dtype=np.int64)                         # Number of late events for multiwire data
        header['FILENAM'] = ['unknown.sfrm']                                      # (Original) frame filename
        header['CREATED'] = ['01-Jan-2000 01:01:01']                              # Date and time of creation
        header['CUMULAT'] = np.array([20.0], dtype=np.float64)                    # Accumulated exposure time in real hours
        header['ELAPSDR'] = np.array([10.0, 10.0], dtype=np.float64)              # Requested time for this frame in seconds
        header['ELAPSDA'] = np.array([10.0, 10.0], dtype=np.float64)              # Actual time for this frame in seconds
        header['OSCILLA'] = np.array([0], dtype=np.int64)                         # Nonzero if acquired by oscillation
        header['NSTEPS']  = np.array([1], dtype=np.int64)                         # steps or oscillations in this frame
        header['RANGE']   =  np.array([1.0], dtype=np.float64)                    # Magnitude of scan range in decimal degrees
        header['START']   = np.array([0.0], dtype=np.float64)                     # Starting scan angle value, decimal deg
        header['INCREME'] = np.array([1.0], dtype=np.float64)                     # Signed scan angle increment between frames
        header['NUMBER']  = np.array([1], dtype=np.int64)                         # Number of this frame in series (zero-based)
        header['NFRAMES'] = np.array([1], dtype=np.int64)                         # Number of frames in the series
        header['ANGLES']  = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
        header['NOVER64'] = np.array([0, 0, 0], dtype=np.int64)                   # Number of pixels > 64K
        header['NPIXELB'] = np.array([1, 2], dtype=np.int64)                      # Number of bytes/pixel; Number of bytes per underflow entry
        header['NROWS']   = np.array([512, 1], dtype=np.int64)                    # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
                                                                                # for each mosaic tile, X varying fastest
        header['NCOLS']   = np.array([512, 1], dtype=np.int64)                    # Number of pixels per row; number of mosaic tiles in X; dZ/dX
                                                                                # value for each mosaic tile, X varying fastest
        header['WORDORD'] = np.array([0], dtype=np.int64)                         # Order of bytes in word; always zero (0=LSB first)
        header['LONGORD'] = np.array([0], dtype=np.int64)                         # Order of words in a longword; always zero (0=LSW first
        header['TARGET']  = ['Mo']                                                # X-ray target material)
        header['SOURCEK'] = np.array([0.0], dtype=np.float64)                     # X-ray source kV
        header['SOURCEM'] = np.array([0.0], dtype=np.float64)                     # Source milliamps
        header['FILTER']  = ['?']                                                 # Text describing filter/monochromator setting
        header['CELL']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Cell constants, 2 lines  (A,B,C,Alpha,Beta,Gamma)
        header['MATRIX']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Orientation matrix, 3 lines
        header['LOWTEMP'] = np.array([1, -17300, -6000], dtype=np.int64)          # Low temp flag; experiment temperature*100; detector temp*100
        header['ZOOM']    = np.array([0.0, 0.0, 1.0], dtype=np.float64)           # Image zoom Xc, Yc, Mag
        header['CENTER']  = np.array([256.0, 256.0, 256.0, 256.0], dtype=np.float64) # X, Y of direct beam at 2-theta = 0
        header['DISTANC'] = np.array([5.0], dtype=np.float64)                     # Sample-detector distance, cm
        header['TRAILER'] = np.array([0], dtype=np.int64)                         # Byte pointer to trailer info (unused; obsolete)
        header['COMPRES'] = ['none']                                              # Text describing compression method if any
        header['LINEAR']  = np.array([1.0, 0.0], dtype=np.float64)                # Linear scale, offset for pixel values
        header['PHD']     = np.array([0.0, 0.0], dtype=np.float64)                # Discriminator settings
        header['PREAMP']  = np.array([0], dtype=np.int64)                         # Preamp gain setting
        header['CORRECT'] = ['UNKNOWN']                                           # Flood correction filename
        header['WARPFIL'] = ['UNKNOWN']                                           # Spatial correction filename
        header['WAVELEN'] = np.array([0.1, 0.1, 0.1], dtype=np.float64)           # Wavelengths (average, a1, a2)
        header['MAXXY']   = np.array([1, 1], dtype=np.int64)                      # X,Y pixel # of maximum counts
        header['AXIS']    = np.array([2], dtype=np.int64)                         # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
        header['ENDING']  = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float64)      # Setting angles read at end of scan
        header['DETPAR']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
        header['LUT']     = ['lut']                                               # Recommended display lookup table
        header['DISPLIM'] = np.array([0.0, 0.0], dtype=np.float64)                # Recommended display contrast window settings
        header['PROGRAM'] = ['Python Image Conversion']                           # Name and version of program writing frame
        header['ROTATE']  = np.array([0], dtype=np.int64)                         # Nonzero if acquired by rotation (GADDS)
        header['BITMASK'] = ['$NULL']                                             # File name of active pixel mask (GADDS)
        header['OCTMASK'] = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)    # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
        header['ESDCELL'] = np.array([0.001, 0.001, 0.001, 0.02, 0.02, 0.02], dtype=np.float64) # Cell ESD's, 2 lines (A,B,C,Alpha,Beta,Gamma)
        header['DETTYPE'] = ['Unknown']                                           # Detector type
        header['NEXP']    = np.array([1, 0, 0, 0, 0], dtype=np.int64)             # Number exposures in this frame; CCD bias level*100,;
                                                                                # Baseline offset (usually 32); CCD orientation; Overscan Flag
        header['CCDPARM'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
        header['CHEM']    = ['?']                                                 # Chemical formula
        header['MORPH']   = ['?']                                                 # CIFTAB string for crystal morphology
        header['CCOLOR']  = ['?']                                                 # CIFTAB string for crystal color
        header['CSIZE']   = ['?']                                                 # String w/ 3 CIFTAB sizes, density, temp
        header['DNSMET']  = ['?']                                                 # CIFTAB string for density method
        header['DARK']    = ['NONE']                                              # Dark current frame name
        header['AUTORNG'] = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64) # Autorange gain, time, scale, offset, full scale
        header['ZEROADJ'] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Adjustments to goniometer angle zeros (tth, omg, phi, chi)
        header['XTRANS']  = np.array([0.0, 0.0, 0.0], dtype=np.float64)           # Crystal XYZ translations
        header['HKL&XY']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # HKL and pixel XY for reciprocal space (GADDS)
        header['AXES2']   = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Diffractometer setting linear axes (4 ea) (GADDS)
        header['ENDING2'] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Actual goniometer axes @ end of frame (GADDS)
        header['FILTER2'] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)      # Monochromator 2-theta, roll (both deg)
        header['LEPTOS']  = ['']
        header['CFR']     = ['']
        return header
        
    def write_bruker_frame(fname, fheader, fdata):
        ########################
        ## write_bruker_frame ##
        ##     FUNCTIONS      ##
        ########################
        def pad_table(table, bpp):
            '''
            pads a table with zeros to a multiple of 16 bytes
            '''
            padded = np.zeros(int(np.ceil(table.size * abs(bpp) / 16)) * 16 // abs(bpp)).astype(_BPP_TO_DT[bpp])
            padded[:table.size] = table
            return padded
            
        def format_bruker_header(fheader):
            '''
            
            '''
            format_dict = {(1,   'int64'): '{:<71d} ',
                        (2,   'int64'): '{:<35d} {:<35d} ',
                        (3,   'int64'): '{:<23d} {:<23d} {:<23d} ',
                        (4,   'int64'): '{:<17d} {:<17d} {:<17d} {:<17d} ',
                        (5,   'int64'): '{:<13d} {:<13d} {:<13d} {:<13d} {:<13d}   ',
                        (6,   'int64'): '{:<11d} {:<11d} {:<11d} {:<11d} {:<11d} {:<11d} ',
                        (1,   'int32'): '{:<71d} ',
                        (2,   'int32'): '{:<35d} {:<35d} ',
                        (3,   'int32'): '{:<23d} {:<23d} {:<23d} ',
                        (4,   'int32'): '{:<17d} {:<17d} {:<17d} {:<17d} ',
                        (5,   'int32'): '{:<13d} {:<13d} {:<13d} {:<13d} {:<13d}   ',
                        (6,   'int32'): '{:<11d} {:<11d} {:<11d} {:<11d} {:<11d} {:<11d} ',
                        (1, 'float64'): '{:<71f} ',
                        (2, 'float64'): '{:<35f} {:<35f} ',
                        (3, 'float64'): '{:<23f} {:<23f} {:<23f} ',
                        (4, 'float64'): '{:<17f} {:<17f} {:<17f} {:<17f} ',
                        (5, 'float64'): '{:<13f} {:<13f} {:<13f} {:<13f} {:<15f} '}
        
            headers = []
            for name, entry in fheader.items():
                # TITLE has multiple lines
                if name == 'TITLE':
                    name = '{:<7}:'.format(name)
                    number = len(entry)
                    for line in range(8):
                        if number < line:
                            headers.append(''.join((name, '{:<72}'.format(entry[line]))))
                        else:
                            headers.append(''.join((name, '{:<72}'.format(' '))))
                    continue
        
                # DETTYPE Mixes Entry Types
                if name == 'DETTYPE':
                    name = '{:<7}:'.format(name)
                    string = '{:<20s} {:<11f} {:<11f} {:<1d} {:<11f} {:<10f} {:<1d} '.format(*entry)
                    headers.append(''.join((name, string)))
                    continue
                
                # format the name
                name = '{:<7}:'.format(name)
                
                # pad entries
                if type(entry) == list or type(entry) == str:
                    headers.append(''.join(name + '{:<72}'.format(entry[0])))
                    continue
                
                # fill empty fields
                if entry.shape[0] == 0:
                    headers.append(name + '{:72}'.format(' '))
                    continue
                
                # if line has too many entries e.g.
                # OCTMASK(8): np.int64
                # CELL(6), MATRIX(9), DETPAR(6), ESDCELL(6): np.float64
                # write the first 6 (np.int64) / 5 (np.float64) entries
                # and the remainder in a new line/entry
                if entry.shape[0] > 6 and entry.dtype == np.int64:
                    while entry.shape[0] > 6:
                        format_string = format_dict[(6, str(entry.dtype))]
                        headers.append(''.join(name + format_string.format(*entry[:6])))
                        entry = entry[6:]
                elif entry.shape[0] > 5 and entry.dtype == np.float64:
                    while entry.shape[0] > 5:
                        format_string = format_dict[(5, str(entry.dtype))]
                        headers.append(''.join(name + format_string.format(*entry[:5])))
                        entry = entry[5:]
                
                # format line
                format_string = format_dict[(entry.shape[0], str(entry.dtype))]
                headers.append(''.join(name + format_string.format(*entry)))
        
            # add header ending
            if headers[-1][:3] == 'CFR':
                headers = headers[:-1]
            padding = 512 - (len(headers) * 80 % 512)
            end = '\x1a\x04'
            if padding <= 80:
                start = 'CFR: HDR: IMG: '
                padding -= len(start) + 2
                dots = ''.join(['.'] * padding)
                headers.append(start + dots + end)
            else:
                while padding > 80:
                    headers.append(end + ''.join(['.'] * 78))
                    padding -= 80
                if padding != 0:
                    headers.append(end + ''.join(['.'] * (padding - 2)))
            return ''.join(headers)
        ########################
        ## write_bruker_frame ##
        ##   FUNCTIONS END    ##
        ########################
        
        # assign bytes per pixel to numpy integers
        # int8   Byte (-128 to 127)
        # int16  Integer (-32768 to 32767)
        # int32  Integer (-2147483648 to 2147483647)
        # uint8  Unsigned integer (0 to 255)
        # uint16 Unsigned integer (0 to 65535)
        # uint32 Unsigned integer (0 to 4294967295)
        _BPP_TO_DT = {1: np.uint8,
                    2: np.uint16,
                    4: np.uint32,
                    -1: np.int8,
                    -2: np.int16,
                    -4: np.int32}
        
        # read the bytes per pixel
        # frame data (bpp), underflow table (bpp_u)
        bpp, bpp_u = fheader['NPIXELB']
        
        # generate underflow table
        # does not work as APEXII reads the data as uint8/16/32!
        if fheader['NOVERFL'][0] >= 0:
            data_underflow = fdata[fdata <= 0]
            fheader['NOVERFL'][0] = data_underflow.shape[0]
            table_underflow = pad_table(data_underflow, -1 * bpp_u)
            fdata[fdata < 0] = 0
    
        # generate 32 bit overflow table
        if bpp < 4:
            data_over_uint16 = fdata[fdata >= 65535]
            table_data_uint32 = pad_table(data_over_uint16, 4)
            fheader['NOVERFL'][2] = data_over_uint16.shape[0]
            fdata[fdata >= 65535] = 65535
    
        # generate 16 bit overflow table
        if bpp < 2:
            data_over_uint8 = fdata[fdata >= 255]
            table_data_uint16 = pad_table(data_over_uint8, 2)
            fheader['NOVERFL'][1] = data_over_uint8.shape[0]
            fdata[fdata >= 255] = 255
    
        # shrink data to desired bpp
        fdata = fdata.astype(_BPP_TO_DT[bpp])
        
        # write frame
        with open(fname, 'wb') as brukerFrame:
            brukerFrame.write(format_bruker_header(fheader).encode('ASCII'))
            brukerFrame.write(fdata.tobytes())
            if fheader['NOVERFL'][0] >= 0:
                brukerFrame.write(table_underflow.tobytes())
            if bpp < 2 and fheader['NOVERFL'][1] > 0:
                brukerFrame.write(table_data_uint16.tobytes())
            if bpp < 4 and fheader['NOVERFL'][2] > 0:
                brukerFrame.write(table_data_uint32.tobytes())
    
    # Goni is vertical, rotate by 90
    idat = np.rot90(idat, k=1, axes=(1, 0))
    
    # Adjust beam center x/y to the rotation
    det_bcx = y
    det_bcy = 1028 - x
    
    # set bad pixels to zero
    idat[idat == det_max] = 0
    
    # scan parameters
    # increment, exposure time, start and end angle of the omega scan
    scn_inc = iswp*isum
    scn_exp = iexp*isum
    scn_end = inum*scn_inc
    scn_sta = scn_end - scn_inc

    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # Eiger2-1M pixel size is 0.075 mm 
    pix_per_512 = round((10.0 / 0.075) * (512.0 / (np.sum(idat.shape) / 2.0)), 6)
    
    # default bruker header
    header = bruker_header()
    
    # fill known header items
    header['NCOLS']      = [idat.shape[1]]                                      # Number of pixels per row; number of mosaic tiles in X; dZ/dX
    header['NROWS']      = [idat.shape[0]]                                      # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
    header['CENTER'][:]  = [det_bcx, det_bcy, det_bcx, det_bcy]                 # adjust the beam center for the filling/cutting of the frame
    header['CCDPARM'][:] = [1.00, 1.00, 1.00, 1.00, det_max]
    header['DETPAR'][:]  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    header['DETTYPE'][:] = ['EIGER2CdTe-1M', pix_per_512, 0.00, 0, 0.001, 0.0, 0]
    header['SITE']       = ['SPring-8/BL02B1']                                  # Site name
    header['MODEL']      = ['Synchrotron']                                      # Diffractometer model
    header['TARGET']     = ['Bending Magnet']                                   # X-ray target material)
    header['USER']       = ['USER']                                             # Username
    header['SOURCEK']    = ['?']                                                # X-ray source kV
    header['SOURCEM']    = ['?']                                                # Source milliamps
    header['WAVELEN'][:] = [src_wav, src_wav, src_wav]                          # Wavelengths (average, a1, a2)
    header['FILENAM']    = ['?']
    header['CUMULAT']    = [scn_exp]                                            # Accumulated exposure time in real hours
    header['ELAPSDR']    = [scn_exp]                                            # Requested time for this frame in seconds
    header['ELAPSDA']    = [scn_exp]                                            # Actual time for this frame in seconds
    header['START'][:]   = scn_sta                                              # Starting scan angle value, decimal deg
    header['ANGLES'][:]  = [gon_tth, scn_sta, gon_phi, gon_chi]                 # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['ENDING'][:]  = [gon_tth, scn_end, gon_phi, gon_chi]                 # Setting angles read at end of scan
    header['TYPE']       = ['Generic Omega Scan']                               # String indicating kind of data in the frame
    header['DISTANC']    = [float(gon_dxt) / 10.0]                              # Sample-detector distance, cm
    header['RANGE']      = [abs(scn_inc)]                                       # Magnitude of scan range in decimal degrees
    header['INCREME']    = [scn_inc]                                            # Signed scan angle increment between frames
    header['NUMBER']     = ['?']                                                # Number of this frame in series (zero-based)
    header['NFRAMES']    = ['?']                                                # Number of frames in the series
    header['AXIS'][:]    = [2]                                                  # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['LOWTEMP'][:] = [1, int((-273.15 + 20.0) * 100.0), -6000]            # Low temp flag; experiment temperature*100; detector temp*100
    header['NEXP'][2]    = 0
    header['MAXXY']      = np.array(np.where(idat == idat.max()), np.float)[:, 0]
    header['MAXIMUM']    = [int(np.max(idat))]
    header['MINIMUM']    = [int(np.min(idat))]
    header['NCOUNTS'][:] = [idat.sum(), 0]
    header['NOVER64'][:] = [idat[idat > 64000].shape[0], 0, 0]
    header['NSTEPS']     = [1]                                                  # steps or oscillations in this frame
    header['NPIXELB'][:] = [1, 1]                                               # bytes/pixel in main image, bytes/pixel in underflow table
    header['COMPRES']    = ['NONE']                                             # compression scheme if any
    header['TRAILER']    = [0]                                                  # byte pointer to trailer info
    header['LINEAR'][:]  = [1.00, 0.00]     
    header['PHD'][:]     = [1.00, 0.00]
    header['OCTMASK'][:] = [0, 0, 0, 1023, 1023, 2046, 1023, 1023]
    header['DISPLIM'][:] = [0.0, 100.0]                                         # Recommended display contrast window settings
    header['FILTER2'][:] = [90.0, 0.0, 0.0, 1.0]                                # Monochromator 2-theta, roll (both deg)
    header['CREATED']    = ['?']                                                # use creation time of raw data!
    
    # write the frame
    write_bruker_frame(fnam, header, idat)
    return True

def parallel_read(set, idx, inum, _ARGS):
    # open the h5 file and sum the images in the given range
    with h5py.File(_ARGS._H5F, 'r') as h5:
        arr2d = np.sum(h5['entry/data'][set][idx:idx+_ARGS._SUM,:,:], axis=0)
    # figure out a name
    inam = '{}_{:02}_{:04}.sfrm'.format(os.path.join(_ARGS._OUT, _ARGS._NAM), _ARGS._RUN, inum)
    # convert the summed 2d array to Bruker sfrm
    # if successful return True else False
    if convert_SP8_Eiger2_Bruker(inam, inum, _ARGS._SUM, _ARGS._FET, _ARGS._OSR, arr2d, src_wav=_ARGS._WAV, det_max=_ARGS._IBD*_ARGS._SUM, gon_chi=_ARGS._CHI, gon_tth=_ARGS._TTH):
        return True
    else:
        return False
    
if __name__ == '__main__':
    '''
     Script to convert Eiger2 CdTe 1M data collected at SPring-8/BL02B1 to Bruker .sfrm format
     
     The goniometer angles Chi and 2-Theta for the given run have to be provided
     as arguments as these are unavailable from the h5 files metadata
      -c: Chi angle
      -t: 2-Theta angle
      -r: Run number
     
     The omega scan range (in degrees per image) has to be specified
      -a: Omega scan range deg/image
      
     The wavelength stored if the h5 file is inaccuarate as it is the
     set energy threshhold and has to be provided as well
      -w: Wavelength
     
     Images can be combined to reduce the number of sfrm files
      -s: Number of images to sum
      
      - Exposure time and scan range is adjusted accordingly
      - The slice or number of combined images has to be a multiple
        of the total number of images
    
     Info read from the metadata:
      - the frametime (exposure time per image)
      - image bit depth (to identify bad pixels)
      - x and y pixel dimensions of the detector
     
     Parallelizing the conversion:
      - can't pickle the .h5 file, but
      - multiple readers are allowed
      - multiprocessing pool.apply_async to parallelize, where
        every spawned process accesses the h5 file,
        reads a slice (_ARGS._SUM) of data, summes up the arrays
        and converts the resulting array to sfrm format
      - not I/O bound because of compression?
      - speedup increases roughly linear with the CPU number!
    '''
    # interpret the arguments
    _ARGS = init_argparser()
    
    # create the output directory
    if not os.path.exists(_ARGS._OUT):
        os.mkdir(_ARGS._OUT)
    
    # read the h5 file and get some info
    with h5py.File(_ARGS._H5F, 'r') as h5:
        # frame exposure time
        _ARGS._FET = round(h5.get('entry/instrument/detector/frame_time')[()], 6)
        # image bit depth
        _ARGS._IBD  = 2**h5.get('entry/instrument/detector/bit_depth_image')[()] -1
        # beam center x/y
        _ARGS._BCX = h5.get('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector')[()]
        _ARGS._BCY = h5.get('/entry/instrument/detector/detectorSpecific/y_pixels_in_detector')[()]
        # the h5 files we need to iterate
        h5files = list(h5['entry/data'].keys())
        # check if the files are available or None
        h5data = [h5['entry/data'].get(i, default=None) for i in h5files]
        # get the number of images in the file (that are available)
        h5inum = [i.len() for i in h5data if i is not None]
        # remove the entries of unavailable files from the list
        h5files = [h5files[i] for i,x in enumerate(h5data) if x is not None]
    
    # check if the image number of images stored in the individual files are a
    # multiple of the number of images to sum
    if all([i % _ARGS._SUM for i in h5inum]):
        print('ERROR: Individual number of images {} is not compatible with {}!'.format(h5inum, _ARGS._SUM))
        raise SystemExit
    # check if total number of images is a multiple of the desired sum
    if sum(h5inum) % _ARGS._SUM != 0:
        print('ERROR: Total number of images ({}) is not a multiple of {}!'.format(sum(h5inum), _ARGS._SUM))
        raise SystemExit
    
    # todo: total number of images to be converted
    todo = sum(h5inum) // _ARGS._SUM
    # list containing successful conversions: True or False
    converted = []
    # we use a pool of workers to access slices (_ARGS._SUM) of the data array
    # in parallel, sum these arrays and convert them
    with mp.Pool() as pool:
        # the number of the resulting sfrm file
        # it is assumed that one h5master file contains one run of data
        # so inum is running continuously over one h5master file
        inum = 0
        # iterate over the h5files, the enumeration (idx) is needed to get the
        # number of images stored in this h5file (h5inum[idx]) to end the while loop
        for ih5, entry in enumerate(h5files):
            # index to start the current slice and slice from start to start + _ARGS._SUM
            # the data (arrays) of one slice are summed and converted to sfrm
            idx = 0
            # as long as idx + _ARGS._SUM is smaller than the number of images
            # in h5inum[ih5] there is work to do
            while idx + _ARGS._SUM < h5inum[ih5]:
                # increment the sfrm image number
                inum += 1
                # run the conversion of the slice in parallel
                pool.apply_async(parallel_read, args=[entry,idx,inum,_ARGS], callback=converted.append)
                # increment start
                idx += _ARGS._SUM
        # we're done filling the pool
        pool.close()
        # track the progress
        while len(converted) < todo:
            print('> {:5.1f}% ({:{w}}/{:{w}})'.format(float(len(converted)) / float(todo) * 100.0, len(converted), todo, w=len(str(todo))), end='\r')
        # the processes should be done already, if not let's wait
        pool.join()
        # now we're done, count the successful (True) conversions
        print('Successfully converted {} images!'.format(np.count_nonzero(converted)))

