/**
 * @file serial.c
 * @brief File for serial/node/proc IO modes of benchio
 *
 * Contains code relating to reading and writing using standard IO calls.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 *
 * @details
 * The "serial" IO routine takes a communicator argument. This enables
 * it to be used for a variety of purposes:
 * MPI_COMM_WORLD: standard "master" IO from a single process
 * MPI_COMM_NODE:  file-per-node
 * MPI_COMM_SELF:  file-per-process
 */

#include "benchio.h"

/**
 * @brief Perform an write to disk from rank 0 of the specified communicator.
 *
 * Write same amount of data as the parallel write but do it all from rank 0
 * This is just to get a baseline figure for serial IO performance - note
 * that the contents of the file will be different from the parallel calls.
 *
 * @param[in] file_name Name of the file to create
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] communicator Communicator from which to check if rank is 0 and should perform write
 */
void serial_write(char const *file_name, double ***io_data, double *local_sizes, MPI_Comm communicator)
{
    int my_rank, world_size;

    // In serial mode, the communicator passed here is the global one, so only a single process performs the write.
    // In node mode, the communicator is the node-spanning communicator, so one process per mode writes.
    // In proc mode, the communicator is MPI_COMM_SELF, so all processes are rank 0 and will write.

    MPI_Comm_size(communicator, &world_size);
    MPI_Comm_rank(communicator, &my_rank);

    if (my_rank == RANK_ZERO)
    {
        FILE *the_file = fopen(file_name, "wb");

        // Use world_size to determine how much data current process needs to write so that the total data written is the size of the global array
        for (int i = 0; i < world_size; i++)
        {
            fwrite(&io_data[0][0][0], sizeof(double), local_sizes[0] * local_sizes[1] * local_sizes[2], the_file);
        }

        fclose(the_file);
    }
}

/**
 * @brief Perform an read from disk on rank 0 of the specified communicator.
 *
 * Read data back in the same way it was written using serial_write.
 *
 * @param[in] file_name Name of the file to open to read
 * @param[in] io_data 3D array to fill with data from the read
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] communicator Communicator from which to check if rank is 0 and should perform read
 */
void serial_read(char const *file_name, double ***io_data, double *local_sizes, MPI_Comm communicator)
{
    int my_rank, world_size;

    MPI_Comm_size(communicator, &world_size);
    MPI_Comm_rank(communicator, &my_rank);

    // Read same amount of data as the parallel read but do it all from rank 0
    // See serial_write for more detailed explanation

    if (my_rank == RANK_ZERO)
    {
        FILE *the_file = fopen(file_name, "rb");

        for (int i = 0; i < world_size; i++)
        {
            fread(&io_data[0][0][0], sizeof(double), local_sizes[0] * local_sizes[1] * local_sizes[2], the_file);
        }

        fclose(the_file);
    }
}