/**
 * @file benchutil.c
 * @brief Utility functions for benchio program.
 *
 * Contains useful utility functions for benchio, including
 * some functions used for debug mode.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 */

#include <ctype.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

#include "benchio.h"

/**
 * @brief Read data which has been written by the benchmark back and verify its correctness.
 *
 * Use a single process to read the data which was written to file through the benchmark from file,
 * number by number, checking that it forms the correct pattern (a series of incrementing double-precision
 * values starting from 1 and ending at global_sizes[0]*global_sizes[1]*global_sizes[2]). If ADIOS2 was used,
 * call another function to verify since it works differently.
 *
 * @param[in] io_mode The IO mode that was used for the write
 * @param[in] file_name The name of the file/folder to verify correctness of
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] io_comm The global MPI communicator
 */
void verify_output(int io_mode, char *file_name, double *global_sizes, MPI_Comm io_comm)
{
    // Handle ADIOS2 files, which have a different structure depending on underlying mode
    if (io_mode == ADIOS2_IO_IDX)
    {
        adios2_verify(file_name, global_sizes, io_comm);
        return;
    }

    int my_rank;
    MPI_Comm_rank(io_comm, &my_rank);

    if (my_rank == 0)
    {
        FILE *the_file = fopen(file_name, "rb");

        int const one_element = 1;

        double last_value = 0;
        double new_value;

        bool incorrect = false;

        if (the_file == NULL)
        {
            printf("Could not verify correctness of output because the file could not be opened.\n");
            return;
        }

        // Read a double at a time
        while (fread(&new_value, sizeof(double), one_element, the_file) == one_element)
        {
            if (new_value - last_value == 1)
                last_value = new_value;

            else
            {
                incorrect = true;
                break;
            }
        }

        // Check that the last number seen was the expected last number
        if (new_value != global_sizes[0] * global_sizes[1] * global_sizes[2])
            incorrect = true;

        if (incorrect)
        {
            // Rewind the read a little bit to demonstrate the incorrect part

            long file_size = ftell(the_file);

            long offset = (DEBUG_PRINT_NUM_VALUES + 1) * sizeof(double);
            long new_position = file_size - offset;

            // Showing from start of file
            if (new_position < 0)
                rewind(the_file);

            // We have skipped some values
            else
            {
                printf("... ");
                fseek(the_file, -offset, SEEK_CUR);
            }

            bool first = true;

            while (fread(&new_value, sizeof(double), one_element, the_file) == one_element)
            {
                printf("%lf ", new_value);

                if (first)
                {
                    first = false;
                    last_value = new_value;
                }

                else if (new_value - last_value == 1)
                    last_value = new_value;

                else
                    break;
            }

            printf("<-- VALIDATION ERROR OCCURED HERE\n");
        }

        fclose(the_file);
    }
}

/**
 * @brief Verify that data was read back correctly.
 *
 * For the read-based benchmark, verify that each process holds the expected
 * data. Used in debug mode.
 *
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 * @param[in] coords The coordinates of the current process in cartesian_comm
 * @param[in] global_sizes An array of values indicating the local array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 */
void verify_input(MPI_Comm cartesian_comm, double ***io_data, int *coords, double *local_sizes, double *global_sizes)
{
    int my_rank;

    MPI_Comm_rank(cartesian_comm, &my_rank);

    // Run through the same logic as when populating IO data to check that it is correct
    // See populate_io_data in benchio.c for more explanation of the logic

    for (int j_local = 0; j_local < local_sizes[0]; j_local++)
    {
        for (int i_local = 0; i_local < local_sizes[1]; i_local++)
        {
            for (int k_local = 0; k_local < local_sizes[2]; k_local++)
            {
                // Determine the current global position in the array according to current local position in array and process' global position
                double j_global = coords[0] * local_sizes[0] + j_local;
                double i_global = coords[1] * local_sizes[1] + i_local;
                double k_global = coords[2] * local_sizes[2] + k_local;

                /*
                 * Use the global position to calculate current index
                 *
                 * Intuition:
                 * - For each j coordinate, we've skipped global_sizes[1]*global_sizes[2] numbers
                 * - For each i coordinate, we've skipped global_sizes[2] numbers
                 * - Then we just need to add on the value of k
                 * - Finally Add one to the end to make count start at 1
                 */

                if (io_data[j_local][i_local][k_local] != k_global + i_global * global_sizes[2] + j_global * global_sizes[1] * global_sizes[2] + 1)
                {
                    // Could print more detailed information than this, but would risk cluttering the output a lot
                    // and would not necessarily be that understandable (the expected read-back pattern is not as
                    // straight forward as the written pattern)
                    printf("READ-BACK VALIDATION ERROR OCCURRED ON PROCESS %d\n", my_rank);
                    return;
                }
            }
        }
    }
}

/**
 * @brief Check if two strings are the same in a case-insensitive way.
 *
 * Compare two strings character by character to see if they are the same,
 * but convert each character to lowercase before making the comparison.
 *
 * @param[in] string_one First character array (string)
 * @param[in] string_two Second character array (string)
 *
 * @return Return `true` if the two strings were the same (ignoring case) and `false` otherwise.
 */
bool equals_ignore_case(char const *string_one, char const *string_two)
{
    while (*string_one != NULLCHAR && *string_two != NULLCHAR)
    {
        if (tolower(*string_one) != tolower(*string_two))
            return false;

        string_one++;
        string_two++;
    }

    return *string_one == *string_two;
}

/**
 * @brief Convert a char array to an integer
 *
 * Take some char array string number_string and convert it to an integer
 * assuming checks pass. The checks confirm that the user input only contains digits,
 * is over 0, and is within the integer bounds.
 *
 * @param[in] number_string The number, as a character array
 * @param[out] number_int   The number as an integer if applicable
 *
 * @return Return `true` if number successfully parsed, otherwise `false`
 */
bool string_to_integer(char *number_string, int *number_int)
{
    /*
     * This input sanitation ensures a valid, positive, integer was input
     * Which contains nothing else than digits
     */

    char *curr = &number_string[0];
    int length = 0;

    // Check that length is > 0 and only contains digits
    while (*curr != NULLCHAR)
    {
        if (!isdigit(*curr))
            return false;
        curr++;
        length++;
    }

    char *endptr;
    errno = 0;

    // If valid, convert to integer
    if (length > 0)
    {
        int const base = 10;

        // Use strtol to do the conversion as it has better error handling than atoi
        long number_long = strtol(number_string, &endptr, base);

        // Now check that the conversion succeeded and that it is also a valid integer
        if (errno == 0 && length > 0 && number_long > 0 && number_long <= INT_MAX && endptr != number_string)
        {
            *number_int = (int)number_long;
            return true;
        }
    }

    return false;
}

/**
 * @brief Delete the specified file from rank 0 of the passed communicator
 *
 * @param[in] io_mode The IO mode that was used for the write
 * @param[in] file_name The name of the file/folder to delete
 * @param[in] communicator The global MPI communicator
 *
 * @return Returns `true` if the file was succesfully deleted, otherwise `false`
 */
bool boss_delete(enum io_mode io_mode, char const *file_name, MPI_Comm communicator)
{
    int my_rank;
    MPI_Comm_rank(communicator, &my_rank);

    if (my_rank == RANK_ZERO)
    {
        // These modes create folders, so they need special treatment
        if (io_mode == adios_bp3 || io_mode == adios_bp4 || io_mode == adios_bp5)
            return adios2_native_cleanup(file_name, io_mode);

        return remove(file_name) == 0;
    }

    return true;
}

/**
 * @brief Allocate a continuous 3D array of specified dimensions.
 *
 * Dynamically allocate a 3D continuous array of specified type and dimensions.
 * Important: To function correctly, pass `my_array[0][0][0]` to any libraries functions,
 * instead of just `my_array`.
 *
 * @param[in] nx Size (number of elements) of the array to create in the first dimension
 * @param[in] ny Size (number of elements) of the array to create in the second dimension
 * @param[in] nz Size (number of elements) of the array to create in the third dimension
 * @param[in] typesize Size (bytes) of each element in the array
 *
 * @return Pointer to 3D allocated data
 *
 * @details
 * Code provided by David Henty, from: https://github.com/davidhenty/bcastc/
 *
 * @author David Henty
 */
void ***arraymalloc3d(int nx, int ny, int nz, size_t typesize)
{
    void ***array3d;

    size_t it, jt, nxt, nyt, nzt, mallocsize;

    nxt = nx;
    nyt = ny;
    nzt = nz;

    // total memory requirements including pointers

    mallocsize = (nxt + nxt * nyt) * sizeof(void *) + nxt * nyt * nzt * typesize;

    array3d = malloc(mallocsize);

    if (array3d == NULL)
        return NULL;

    for (it = 0; it < nxt; it++)
    {
        array3d[it] = (void **)(array3d + nxt + nyt * it);

        for (jt = 0; jt < nyt; jt++)
        {
            array3d[it][jt] = (void *)(((char *)(array3d + nxt + nxt * nyt)) + (nyt * nzt * it + nzt * jt) * typesize);
        }
    }
    return array3d;
}

/**
 * @brief Print a simple instruction manual to the user
 */
void print_simple_usage()
{
    printf("Usage:\tbenchio -n1 (size) -n2 (size) -n3 (size) (--scale|-sc) (local|global)\n\t");
    printf("[--mode|-m] [serial] [proc] [node] [mpiio] [hdf5] [netcdf] [adios]\n\t");
    printf("[--stripe|-st] [unstriped] [striped] [fullstriped] [--read|-r]\n");
    printf("For help, run benchio --help\n");
}

/**
 * @brief Print detailed instructions of how to use benchio to the user
 */
void print_detailed_usage()
{
    printf("\nUsage:\tbenchio -n1 (size) -n2 (size) -n3 (size) (--scale|-sc) (local|global) [options]\n");
    printf("Run a (writing-based) benchmark with an array of size n1 x n2 x n3 on each process (local) or across all processes (global).\n\n");
    printf("If no IO mode or stripe mode is provided, then all modes are enabled.\n\n");

    printf("Options:\n");

    // Required flags
    printf("  -n1, --n1                     Size in the first dimension (required)\n");
    printf("  -n2, --n2                     Size in the second dimension (required)\n");
    printf("  -n3, --n3                     Size in the third dimension (required)\n");

    printf("  -sc, --scale (local|global)   Specify whether the size is specified locally per process or globally across all processes (required)\n");

    // Optional flags
    printf("  -h, --help                    Display these instructions and exit (optional)\n");
    printf("  -m,  --mode (mode)            Which IO modes to use (optional). One or more of [serial] [proc] [node] [mpiio] [hdf5] [netcdf] [adios]\n");
    printf("  -r, --read                    Record read-back times from disk after writing data.\n");
    printf("  -st, --stripe (stripe)        Which stripe mode to use (optional). One or more of [unstriped] [striped] [fullstriped]\n");

    printf("\nExamples:\n");
    printf("  benchio -n1 256 -n2 256 -n3 256 -sc local\n");
    printf("  benchio -n1 256 -n2 256 -n3 256 -sc global -m serial mpiio adios\n");

    printf("\nNote:\n");
    printf("You MUST ensure the folders \"unstriped\", \"striped\", and \"fullstriped\" exist and ajust their striping value before running.\n");
    printf("Set unstriped to have a single stripe. In Lustre: lfs setstripe -c 1 unstriped\n");
    printf("Set striped to use an intermediate number of stripes, for example 4. In Lustre: lfs setstripe -c 4 striped\n");
    printf("Set fullstriped to use the maximum number of stripes. In Lustre: lfs setstripe -c -1 fullstriped\n\n");
}