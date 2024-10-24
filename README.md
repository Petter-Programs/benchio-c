# C-based benchio
Simple C parallel IO benchmark for teaching and benchmarking purposes.

This is a ported version of the benchio parallel IO benchmark, which was originally developed in Fortran by EPCC:
[https://github.com/davidhenty/benchio](https://github.com/davidhenty/benchio). This README.md file is modified from the Fortran version's README file.

## Installing benchio

Note that, before running the benchmark, you *must* manually set the striping on the three directories `unstriped`, `striped` and `fullstriped`.

If you are running Lustre (for example on Cirrus and ARCHER2), then these are the instructions to do so:

 * Set `unstriped` to have a single stripe: `lfs setstripe -c 1 unstriped`
 * Set `fullstriped` to use the maximum number of stripes: `lfs setstripe -c -1 fullstriped`
 * Set `striped` to use an intermediate number of stripes, e.g. for 4 stripes: `lfs setstripe -c 4 striped`

If you are running some other filesystem, then check the user guide for that system.

### ARCHER2 Makefile
A sample makefile is available for ARCHER2. You must first load the required modules for ADIOS2 to install; instructions are included inside the Makefile. You can then simply run `make -f Makefile_ARCHER2` to compile. If you desire to compile without ADIOS2, then run `NOADIOS=true make -f Makefile_ARCHER2`.

### Cirrus Makefile
A sample makefile is available for Cirrus. Load MPI (e.g. `module load mpt`) and then use `make -f Makefile_Cirrus` to compile. Note that this will compile without ADIOS2 by default. If you have installed ADIOS2 locally, then you can compile with ADIOS2 using `USEADIOS=true make -f Makefile_Cirrus`

### Other Systems
You will have to alter one of the existing makefiles to suit your needs. Note that benchio is designed such that if the macro `NOADIOS` is defined, then ADIOS2-specific code is excluded from the compilation.

## Running benchio

To run benchio, you must specify the dimensions of the 3D dataset to
write through the `-n1`, `-n2` and `-n3` flags. Additionally, you must
specify whether the sizes provided are to apply per process, or globally, 
the `-sc (local|global)` flag. 

For example, to run using a 256 x 256 x 256 data array on every
process (i.e. weak scaling):
````
benchio -n1 256 -n2 256 -n3 256 -sc local
````
In this case, the total file size will scale with the number of
processes. If run on 8 processes then the total file size would be 1
GiB.

To run using a 256 x 256 x 256 global array (i.e. strong scaling):
````
benchio -n1 256 -n2 256 -n3 256 -sc global
````
In this case, the file size will be 128 MiB regardless of the number
of processes.

By default, benchio only measures write time. To read the file back
immediately after reading and record the time taken, use the `-r` flag.

A 3D cartesian topology p1 x p2 x p3 is created with dimensions
suggested by `MPI_Dims_create()` to create a global 3D array of size
l1 x l2 x l3 where l1 = p1 x n1 etc. The entries of the distributed IO array are set to globally unique
values 1, 2, ... l1xl2xl3 using the normal C ordering.

The code can use seven IO methods, and for each of them can use up to
three directories with different stripings. At the moment, the C version
of benchio only supports the `serial/proc/node/mpiio/adios` options, and will reject
the other options.

All files are deleted immediately after being written to avoid excess
disk usage.

The full
set of options is:
````
benchio -n1 (size) -n2 (size) -n3 (size) (--scale|-sc) (local|global)
        [--mode|-m] [serial] [proc] [node] [mpiio] [hdf5] [netcdf] [adios]
        [--stripe|-st] [unstriped] [striped] [fullstriped]
        [--read|-r]
````

Additionally, `benchio --help` (or `benchio -h`) can be used to get more information on each option.

If `--mode` is not specified, then all the IO modes are used. Similarly,
if `--stripe` is not specified, then the program will use all striping methods.

1. `serial`: Serial IO from one controller process to a single file `serial.dat` using C binary unformatted write with `fopen(..., "wb");`
 2. `proc`: File-per-process with multiple serial IO to *P* files `rankXXXXXX.dat` using C binary unformatted write with `fopen(.., "wb");`
 3. `node`: File-per-node with multiple serial IO to *Nnode* files `nodeXXXXXX.dat` using C binary unformatted write with `fopen(.., "wb");`
 4. `mpiio`: MPI-IO collective IO to a single file `mpiio.dat` using native (i.e. binary) format
 5. `hdf5`: HDF5 collective IO to a single file `hdf5.dat` (Currently unsupported. See Fortran version of benchio.)
 6. `netcdf`: NetCDF collective IO to a single file `netcdf.dat` (Currently unsupported. See Fortran version of benchio.)
 7. `adios`: ADIOS2 collective IO to a directory/file `adios.dat`
    - ADIOS2 aggregator settings can be changed in the `adios2_config.xml` file
 
 Note that the serial part is designed to give a baseline IO rate. For simplicity, and to ensure we write the same amount of data as for the parallel
 methods, rank 0 writes out its
 own local array `size` times in succession. Unlike the parallel IO formats, the contents of the file will therefore *not* be a linearly increasing set of
 values 1, 2, 3, ..., l1xl2xl3.

## Debug Mode

The C version of benchio includes the ability to check correctness of the dataset that is written to or read from disk. To enable this, compile the software with the DEBUG_MODE macro set to true. The sample makefiles also include convenience features: `DEBUG_MODE=true make -f Makefile_ARCHER2`.

## Known Issues

<b>Known Issues in Release 1.0.0:</b>
* ADIOS2 HDF5 mode is disabled in this version of benchio, because it does not work properly
* ADIOS2 read based benchmark is heavily distorted by caching effects

## Documentation

See the subfolder "Doxygen" for a PDF documentation of the software. The file "Doxyfile" is also provided, should you wish to generate your own version of the documentation.
