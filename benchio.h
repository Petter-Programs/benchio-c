/**
 * @file benchio.h
 * @brief Header file for benchio program.
 *
 * Contains readability constants and function headers for
 * the whole of benchio.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 */

#ifndef BENCHIO_H
#define BENCHIO_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/************************** Readability Constants **********************/

/**
 * @def RANK_ZERO
 * @brief Rank 0 constant included for clarity.
 */
#define RANK_ZERO 0

/**
 * @def DIMENSIONS
 * @brief Dimensions of processes and dataset.
 */
#define DIMENSIONS 3

/**
 * @def MAX_FILENAME_LEN
 * @brief Space to allocate for filename char array.
 */
#define MAX_FILENAME_LEN 64

/**
 * @def MAX_SUFFIX_LEN
 * @brief Space to allocate for filename suffix.
 */
#define MAX_SUFFIX_LEN 32

/**
 * @def IO_METHOD_COUNT
 * @brief How many IO methods to accept.
 */
#define IO_METHOD_COUNT 7

/**
 * @def STRIPE_TYPE_COUNT
 * @brief How many striping types to accept.
 */
#define STRIPE_TYPE_COUNT 3

/**
 * @def SERIAL_IO_IDX
 * @brief Index of the 'serial' IO method in io_method_names of benchio.c.
 */
#define SERIAL_IO_IDX 0

/**
 * @def PROC_IO_IDX
 * @brief Index of the 'proc' IO method in io_method_names of benchio.c.
 */
#define PROC_IO_IDX 1

/**
 * @def NODE_IO_IDX
 * @brief Index of the 'node' IO method in io_method_names of benchio.c.
 */
#define NODE_IO_IDX 2

/**
 * @def MPIIO_IO_IDX
 * @brief Index of the 'mpiio' IO method in io_method_names of benchio.c.
 */
#define MPIIO_IO_IDX 3

/**
 * @def ADIOS2_IO_IDX
 * @brief Index of the 'adios' IO method in io_method_names of benchio.c.
 */
#define ADIOS2_IO_IDX 6

/**
 * @def KIB
 * @brief A base-2 kilobyte (a kibibyte, 2^10 bytes).
 */
#define KIB 1024

/**
 * @def MIB
 * @brief A base-2 megabyte (a mibibyte, 2^20 bytes).
 */
#define MIB (KIB * KIB)

/**
 * @def GIB
 * @brief A base-2 gigabyte (a gibibyte, 2^30 bytes).
 */
#define GIB (KIB * MIB)

/**
 * @def NULLCHAR
 * @brief Character which terminates strings.
 */
#define NULLCHAR '\0'

/**
 * @def LONG_ARG
 * @brief If user input starts with this, it indicates a long argument is to be provided.
 */
#define LONG_ARG "--"

/**
 * @def SHORT_ARG
 * @brief If user input starts with this, it indicates a short argument is to be provided.
 */
#define SHORT_ARG "-"

/**
 * @def SEPARATOR_STRING
 * @brief Standard line to print out. Defined to make sure all lines are same length.
 */
#define SEPARATOR_STRING "----------------------------\n"

/******************************** Debug Mode ****************************/

#ifndef DEBUG_MODE

/**
 * @def DEBUG_MODE
 * @brief Whether or not to enable automatic correctness testing (slow for large files).
 *
 * Can use makefile to enable with command "DEBUG_MODE=true make"
 */
#define DEBUG_MODE false

#endif

/**
 * @def DEBUG_PRINT_NUM_VALUES
 * @brief In debug mode, number of values to print out before a detected error.
 */
#define DEBUG_PRINT_NUM_VALUES 5

/******************************** Enums *********************************/

/**
 * @brief Helpful struct to handle the user input passed to the program.
 */
struct argument
{
    char *long_arg;  /**< The long version of the argument */
    char *short_arg; /**<  The short version of the argument */
    bool standalone; /**<  Whether or not the argument needs more input than itself */
    bool optional;   /**<  Whether or not to go ahead if the argument was not provided */
    bool found;      /**<  Parameter used when processing arguments to indicate if the argument was found or not */
    bool complete;   /**<  Parameter used when processing arguments to indicate if the argument should take no more input */
};

/**
 * @brief Used to keep internal track of what io mode is being used.
 *
 * Mostly useful for adios, which can use different underlying io modes with different behavior.
 */
enum io_mode
{
    none,
    serial,
    mpiio,
    adios_hdf5,
    adios_bp3,
    adios_bp4,
    adios_bp5
};

/******************************** benchio.c ****************************/

/**
 * @brief Parse command-line arguments provided by user.
 *
 * See implementation in benchio.c for detailed documentation.
 */
bool process_args(int argc, char **argv, int my_rank, int *sizes, bool *global_flag, bool *use_io_method, bool *use_stripe_method, bool *read_benchmark, bool *user_wants_help);

/**
 * @brief Set up communicators reflecting the nodal environment currently in use.
 *
 * See implementation in benchio.c for detailed documentation.
 */
void setup_nodes(MPI_Comm communicator, int original_rank, MPI_Comm *node_comm, int *node_number);

/**
 * @brief Fill the current process' array with data consistent with the global array.
 *
 * See implementation in benchio.c for detailed documentation.
 */
void populate_io_data(double *local_sizes, int *coords, double *global_sizes, double ***io_data);

/**
 * @brief Start the benchmark, looping through each step according to configuration, and report results.
 *
 * See implementation in benchio.c for detailed documentation.
 */
void main_benchmark_loop(MPI_Comm cartesian_comm, MPI_Comm node_comm, bool *use_stripe_method, bool *use_io_method, double *local_sizes,
                         double *global_sizes, double global_size_gib, bool read_benchmark, int node_num, enum io_mode adios_io_mode,
                         int *coords, double ***io_data);
/**
 * @brief Run the read-based benchmark according to specified configuration.
 *
 * See implementation in benchio.c for detailed documentation.
 */
bool run_read_benchmark(char *file_name, double *local_sizes, double *global_sizes, MPI_Comm io_comm, int my_rank, int io, double global_size_gib, enum io_mode io_mode, MPI_Comm cartesian_comm, int *coords);

/**
 * @brief Run the write-based benchmark according to specified configuration.
 */
bool run_write_benchmark(char *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm io_comm, int my_rank, int io, double global_size_gib, enum io_mode io_mode);

/******************************** benchutil.c **************************/

/**
 * @brief Check if two strings are the same in a case-insensitive way.
 *
 * See implementation in benchutil.c for detailed documentation.
 */
bool equals_ignore_case(char const *string_one, char const *string_two);

/**
 * @brief Convert a char array to an integer
 *
 * See implementation in benchutil.c for detailed documentation.
 */
bool string_to_integer(char *number_string, int *number_int);

/**
 * @brief Allocate a continuous 3D array of specified dimensions.
 *
 * See implementation in benchutil.c for detailed documentation.
 */
void ***arraymalloc3d(int nx, int ny, int nz, size_t typesize);

/**
 * @brief Delete the specified file from rank 0 of the passed communicator
 *
 * See implementation in benchutil.c for detailed documentation.
 */
bool boss_delete(enum io_mode io_mode, char const *file_name, MPI_Comm communicator);

/**
 * @brief Read data which has been written by the benchmark back and verify its correctness.
 *
 * See implementation in benchutil.c for detailed documentation.
 */
void verify_output(int io_mode, char *file_name, double *global_sizes, MPI_Comm io_comm);

/**
 * @brief Verify that data was read back correctly.
 *
 * See implementation in benchutil.c for detailed documentation.
 */
void verify_input(MPI_Comm cartesian_comm, double ***io_data, int *coords, double *local_sizes, double *global_sizes);

/**
 * @brief Print a simple instruction manual to the user
 */
void print_simple_usage();

/**
 * @brief Print detailed instructions of how to use benchio to the user
 */
void print_detailed_usage();

/******************************** serial.c *****************************/

/**
 * @brief Perform an write to disk from rank 0 of the specified communicator.
 *
 * See implementation in serial.c for detailed documentation.
 */
void serial_write(char const *file_name, double ***io_data, double *local_sizes, MPI_Comm communicator);

/**
 * @brief Perform an read from disk on rank 0 of the specified communicator.
 *
 * See implementation in serial.c for detailed documentation.
 */
void serial_read(char const *file_name, double ***io_data, double *local_sizes, MPI_Comm communicator);

/******************************** mpiio.c ******************************/

/**
 * @brief Perform an MPI-IO write of the global array to file.
 *
 * See implementation in mpiio.c for detailed documentation.
 */
void mpiio_write(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm);

/**
 * @brief Perform an MPI-IO read of the global array from file.
 *
 * See implementation in mpiio.c for detailed documentation.
 */
void mpiio_read(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm);

/******************************** adios2.c *****************************/

/**
 * @def ADIOS_CONFIG_FILE
 * @brief Where to read adios configuration from.
 */
#define ADIOS_CONFIG_FILE "adios2_config.xml"

/**
 * @def ADIOS_GLOBAL_ARRAY_VAR
 * @brief What to name the global array variable of adios.
 */
#define ADIOS_GLOBAL_ARRAY_VAR "adios_global_array"

/**
 * @def ADIOS_IO_NAME
 * @brief Name of the IO handler in adios2 to look for from config.
 */
#define ADIOS_IO_NAME "adios_output"

/**
 * @def ADIOS_MODE_HDF5
 * @brief ADIOS engine type string representing HDF5.
 */
#define ADIOS_MODE_HDF5 "hdf5"

/**
 * @def ADIOS_MODE_BP3
 * @brief ADIOS engine type string representing BP3.
 */
#define ADIOS_MODE_BP3 "bp3"

/**
 * @def ADIOS_MODE_BP4
 * @brief ADIOS engine type string representing BP4.
 */
#define ADIOS_MODE_BP4 "bp4"

/**
 * @def ADIOS_MODE_BP5
 * @brief ADIOS engine type string representing BP5.
 */
#define ADIOS_MODE_BP5 "bp5"

/**
 * @def ADIOS_BP3_FILE_COUNT
 * @brief How many extra files generated by ADIOS' BP3 format to delete.
 */
#define ADIOS_BP3_FILE_COUNT 3

/**
 * @def ADIOS_BP4_FILE_COUNT
 * @brief How many extra files generated by ADIOS' BP4 format to delete.
 */
#define ADIOS_BP4_FILE_COUNT 4

/**
 * @def ADIOS_BP5_FILE_COUNT
 * @brief How many extra files generated by ADIOS' BP5 format to delete.
 */
#define ADIOS_BP5_FILE_COUNT 5

/**
 * @brief Perform an ADIOS2 write of the global array to file.
 *
 * Dummy function if compiled without ADIOS2.
 * See implementation in adios2.c for detailed documentation.
 */
void adios2_write(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm);

/**
 * @brief Perform an ADIOS2 read of the global array from file.
 *
 * Dummy function if compiled without ADIOS2.
 * See implementation in adios2.c for detailed documentation.
 */
void adios2_read(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm);

/**
 * @brief Read data which has been written with ADIOS2 back and verify its correctness.
 *
 * Dummy function if compiled without ADIOS2.
 * See implementation in adios2.c for detailed documentation.
 */
void adios2_verify(char *file_name, double *global_sizes, MPI_Comm communicator);

/**
 * @brief Retreive the current ADIOS2 IO mode in use.
 *
 * Dummy function if compiled without ADIOS2 with a warning printout if called.
 * See implementation in adios2.c for detailed documentation.
 */
bool get_adios2_io_mode(MPI_Comm io_comm, enum io_mode *adios_io_mode);

/**
 * @brief Clean up the files written by the native ADIOS modes.
 *
 * Dummy function if compiled without ADIOS2.
 * See implementation in adios2.c for detailed documentation.
 */
bool adios2_native_cleanup(char const *file_name, enum io_mode io_mode);

#endif