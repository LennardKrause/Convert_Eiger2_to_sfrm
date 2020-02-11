# Convert_Eiger2_to_sfrm (early beta)

Script to convert Dectris Eiger2 CdTe 1M HDF5 (.h5) data to Bruker (.sfrm) format and prepare SAINT (Bruker SAINT+ Integration Engine V8.35A) X-ray Aperture (xa) integration masks

It is currently designed to be used with data collected at [SPring-8](http://www.spring8.or.jp/en/)/[BL02B1](http://www.spring8.or.jp/wkg/BL02B1/instrument/lang-en/INS-0000001275/instrument_summary_view)

 #### Addition: Convert_Eiger2_to_sfrm_RunHack
 sloppy workaround to handle master.h5 files that contain more than one run
 _ARGS._RUNLIST = [(chi angle, 2-theta, run number), (...), ...]
 _ARGS._RUNLIST = [( 0.000,   0.000,  1), ( 0.000,   0.000,  1), (15.000,   0.000,  2), (15.000,   0.000,  2)]

## Important
 - This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY
 - Bruker is not associated with this software and will not support this
 - Dectris is not associated with this software and will not support this

## Requirements

#### [Python](https://www.python.org/) 3.5 or later

#### Libraries (tested with):
 - [numpy (1.16.3)](https://www.numpy.org/)
 - [h5py (2.10.0)](https://www.h5py.org/)
 - hdf5plugin (2.1.1)
