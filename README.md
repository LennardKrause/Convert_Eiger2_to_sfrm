# Convert_Eiger2_to_sfrm

##### Script to convert Eiger2 CdTe 1M data collected at SPring-8/BL02B1 to Bruker .sfrm format

##### Chi and 2-Theta for the given run have to be provided as arguments as these are unavailable from the h5 files metadata
 - _ARGS._CHI: Chi angle
 - _ARGS._TTH: 2-Theta angle

##### The omega scan range (in degrees per image) has to be specified as well
 - _ARGS._OSR: Omega scan range deg/image
 
##### The wavelength stored if the h5 file is inaccuarate as it is the set energy threshhold and has to be provided as well
 - _ARGS._WAV: Wavelength in Ang.
  
##### Info read from the metadata:
 - the frametime (exposure time per image)
 - image bit depth (to identify bad pixels)
 - x and y pixel dimensions of the detector
   
##### Parallelizing the conversion:
 - can't pickle the .h5 file, but
 - multiple readers are allowed
 - multiprocessing pool.apply_async to parallelize, where
   every spawned process accesses the h5 file,
   reads a slice (_ARGS._SUM) of data, summes up the arrays
   and converts the resulting array to sfrm format
 - not I/O bound because of compression?
 - speedup increases roughly linear with the number of CPU!
