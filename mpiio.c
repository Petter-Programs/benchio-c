/**
 * @file mpiio.c
 * @brief File for MPI-IO related code in benchio.
 *
 * Contains code relating to reading and writing MPI-IO files.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 */

#include "benchio.h"

/**
 * @brief Perform an MPI-IO write of the global array to file.
 *
 * Write the global array, stored as a set of local arrays in `io_data`, to the specified
 * file, using MPI-IO. Perform error error checking throughout to report any issues encountered.
 *
 * @param[in] file_name Name of the MPI-IO file to create
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 */
void mpiio_write(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    int my_rank;

    // Local and global sizes are cast to doubles to make calculations that might need more than integer
    // precision not need a cast. However, MPI needs them to be integers, so convert them here.

    int local_sizes_int[DIMENSIONS] = {(int)local_sizes[0], (int)local_sizes[1], (int)local_sizes[2]};
    int global_sizes_int[DIMENSIONS] = {(int)global_sizes[0], (int)global_sizes[1], (int)global_sizes[2]};

    MPI_Comm_rank(cartesian_comm, &my_rank);

    int dims[DIMENSIONS];
    int periods[DIMENSIONS];
    int coords[DIMENSIONS];

    int const total_size = local_sizes[0] * local_sizes[1] * local_sizes[2];

    MPI_Cart_get(cartesian_comm, DIMENSIONS, dims, periods, coords);

    int starts[DIMENSIONS] = {coords[0] * local_sizes[0], coords[1] * local_sizes[1], coords[2] * local_sizes[2]};

    MPI_Datatype subarray_type;

    // Create a 'fixed-point' subarray, which defines what area - in relation to the global array - this process owns
    MPI_Type_create_subarray(DIMENSIONS, global_sizes_int, local_sizes_int, starts, MPI_ORDER_C, MPI_DOUBLE_PRECISION, &subarray_type);

    MPI_Type_commit(&subarray_type);

    MPI_File file_handle;

    // Open the file and attach to file_handle. Explicitly check for errors here, because MPI default is to ignore
    int error = MPI_File_open(cartesian_comm, file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file_handle);

    // Set the MPIIO file view using the subarray that was just created, using native format; implying just writing unformatted binary data
    if (error == MPI_SUCCESS)
        error = MPI_File_set_view(file_handle, 0, MPI_DOUBLE_PRECISION, subarray_type, "native", MPI_INFO_NULL);

    // Collective MPIIO write operation. Just write total_size double-precision elements here, as the view has already been set, so MPIIO will use that pattern
    if (error == MPI_SUCCESS)
        error = MPI_File_write_all(file_handle, &io_data[0][0][0], total_size, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE);

    if (error != MPI_SUCCESS && my_rank == RANK_ZERO)
        printf("WARNING: MPI-IO error occurred during write, results may not be correct\n");

    error = MPI_File_close(&file_handle);

    if (error != MPI_SUCCESS && my_rank == RANK_ZERO)
        printf("MPI-IO error ocurred when closing file\n");

    MPI_Type_free(&subarray_type);
}

/**
 * @brief Perform an MPI-IO read of the global array from file.
 *
 * This function reads data from a file using the MPI-IO library. It reads the data
 * into the `io_data` array using the specified file name, local and global sizes,
 * and MPI communicator.
 *
 * @param[in] file_name Name of the (MPI-IO / binary) file to read from
 * @param[in, out] io_data The pre-allocated current process' 3D array to fill with data from file
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 */
void mpiio_read(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    int my_rank;

    // Local and global sizes are cast to doubles to make calculations that might need more than integer
    // precision not need a cast. However, MPI needs them to be integers, so convert them here.

    int local_sizes_int[DIMENSIONS] = {(int)local_sizes[0], (int)local_sizes[1], (int)local_sizes[2]};
    int global_sizes_int[DIMENSIONS] = {(int)global_sizes[0], (int)global_sizes[1], (int)global_sizes[2]};

    MPI_Comm_rank(cartesian_comm, &my_rank);

    int dims[DIMENSIONS];
    int periods[DIMENSIONS];
    int coords[DIMENSIONS];

    int const total_size = local_sizes[0] * local_sizes[1] * local_sizes[2];

    MPI_Cart_get(cartesian_comm, DIMENSIONS, dims, periods, coords);

    int starts[DIMENSIONS] = {coords[0] * local_sizes[0], coords[1] * local_sizes[1], coords[2] * local_sizes[2]};

    MPI_Datatype subarray_type;

    // Create a 'fixed-point' subarray, which defines what area - in relation to the global array - this process owns
    MPI_Type_create_subarray(DIMENSIONS, global_sizes_int, local_sizes_int, starts, MPI_ORDER_C, MPI_DOUBLE_PRECISION, &subarray_type);

    MPI_Type_commit(&subarray_type);

    MPI_File file_handle;

    // Open the file and attach to file_handle. Explicitly check for errors here, because MPI default is to ignore
    int error = MPI_File_open(cartesian_comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);

    // Set the MPIIO file view using the subarray that was just created, using native format; implying just writing unformatted binary data
    if (error == MPI_SUCCESS)
        error = MPI_File_set_view(file_handle, 0, MPI_DOUBLE_PRECISION, subarray_type, "native", MPI_INFO_NULL);

    // Collective MPIIO write operation. Just write total_size double-precision elements here, as the view has already been set, so MPIIO will use that pattern
    if (error == MPI_SUCCESS)
        error = MPI_File_read_all(file_handle, &io_data[0][0][0], total_size, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE);

    if (error != MPI_SUCCESS && my_rank == RANK_ZERO)
        printf("WARNING: MPI-IO error occurred during read, results may not be correct\n");

    error = MPI_File_close(&file_handle);

    if (error != MPI_SUCCESS && my_rank == RANK_ZERO)
        printf("MPI-IO error ocurred when closing file\n");

    MPI_Type_free(&subarray_type);
}