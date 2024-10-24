/**
 * @file adios2.c
 * @brief File for ADIOS2 related code in benchio.
 *
 * Contains code relating to reading, writing, verifying correctness of,
 * and deleting ADIOS2 files.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 */

#include "benchio.h"

#ifndef NOADIOS

#include <adios2_c.h>

/**
 * @brief File names of each ADIOS2 bp3 native binary format mode, to delete when cleaning up.
 */
char *const bp3_file_names[] = {".bp", ".bp.dir/adios.dat.bp.%d", ".bp.dir/profiling.json"};

/**
 * @brief File names of each ADIOS2 bp4 native binary format mode, to delete when cleaning up.
 */
char *const bp4_file_names[] = {"/data.%d", "/md.%d", "/md.idx", "/profiling.json"};

/**
 * @brief File names of each ADIOS2 bp5 native binary format mode, to delete when cleaning up.
 */
char *const bp5_file_names[] = {"/data.%d", "/md.%d", "/mmd.%d", "/md.idx", "/profiling.json"};

/**
 * @brief Perform an ADIOS2 write of the global array to file.
 *
 * Write the global array, stored as a set of local arrays in `io_data`, to the specified
 * file, using ADIOS2. Perform error error checking throughout to report any issues encountered.
 *
 * @param[in] file_name Name of the ADIOS2 file (folder) to create
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 */
void adios2_write(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    int my_rank;

    MPI_Comm_rank(cartesian_comm, &my_rank);

    int dims[DIMENSIONS];
    int periods[DIMENSIONS];
    int coords[DIMENSIONS];

    MPI_Cart_get(cartesian_comm, DIMENSIONS, dims, periods, coords);

    size_t starts[DIMENSIONS];
    size_t size_t_global[DIMENSIONS];
    size_t size_t_local[DIMENSIONS];

    // User input is already sanitized so these size_t casts are safe (sizes and coords never negative)
    for (int i = 0; i < DIMENSIONS; i++)
    {
        starts[i] = (size_t)(coords[i] * local_sizes[i]);
        size_t_global[i] = (size_t)global_sizes[i];
        size_t_local[i] = (size_t)local_sizes[i];
    }

    // Start ADIOS. Use MPI initialization, and pass the configuration document to ADIOS
    adios2_adios *adios_object = adios2_init_config_mpi(ADIOS_CONFIG_FILE, cartesian_comm);

    // Create an IO handler, to hold the global array
    adios2_io *adios_io = adios2_declare_io(adios_object, ADIOS_IO_NAME);

    // Define the structure of the global array and current position within it
    adios2_variable *global_array_var = adios2_define_variable(adios_io, ADIOS_GLOBAL_ARRAY_VAR, adios2_type_double, DIMENSIONS, size_t_global, starts, size_t_local, adios2_constant_dims_true);

    // Adios engine to execute IO ops
    adios2_engine *adios_engine = adios2_open(adios_io, file_name, adios2_mode_write);

    // Detect and report any issues at this stage
    if (adios_object == NULL || adios_io == NULL || global_array_var == NULL || adios_engine == NULL)
    {
        printf("WARNING: ADIOS2 initialization failed on process %d\n", my_rank);
        adios2_close(adios_engine);
        adios2_finalize(adios_object);
        return;
    }

    adios2_step_status status;

    adios2_begin_step(adios_engine, adios2_step_mode_append, -1, &status);

    adios2_error error = adios2_put(adios_engine, global_array_var, &io_data[0][0][0], adios2_mode_deferred);

    adios2_end_step(adios_engine);

    if (error != adios2_error_none || status != adios2_step_status_ok)
    {
        if (my_rank == RANK_ZERO)
            printf("WARNING: ADIOS2 step did not complete successfully\n");
    }

    adios2_close(adios_engine);
    adios2_finalize(adios_object);
}

/**
 * @brief Perform an ADIOS2 read of the global array from file.
 *
 * This function reads data from a file using the ADIOS2 library. It reads the data
 * into the `io_data` array using the specified file name, local and global sizes,
 * and MPI communicator.
 *
 * @param[in] file_name Name of the ADIOS2 file (folder) to read from
 * @param[in, out] io_data The pre-allocated current process' 3D array to fill with data from file
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 */
void adios2_read(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    int my_rank;

    MPI_Comm_rank(cartesian_comm, &my_rank);

    int dims[DIMENSIONS];
    int periods[DIMENSIONS];
    int coords[DIMENSIONS];

    MPI_Cart_get(cartesian_comm, DIMENSIONS, dims, periods, coords);

    size_t starts[DIMENSIONS];
    size_t size_t_local[DIMENSIONS];

    // User input is already sanitized so these size_t casts are safe (sizes and coords never negative)
    for (int i = 0; i < DIMENSIONS; i++)
    {
        starts[i] = (size_t)(coords[i] * local_sizes[i]);
        size_t_local[i] = (size_t)local_sizes[i];
    }

    // Start ADIOS. Use MPI initialization, and pass the configuration document to ADIOS
    adios2_adios *adios_object = adios2_init_config_mpi(ADIOS_CONFIG_FILE, cartesian_comm);

    // Create an IO handler, to hold the global array
    adios2_io *adios_io = adios2_declare_io(adios_object, ADIOS_IO_NAME);

    // Adios engine to execute IO ops
    adios2_engine *adios_engine = adios2_open(adios_io, file_name, adios2_mode_read);

    // Detect and report any issues at this stage
    if (adios_object == NULL || adios_io == NULL || adios_engine == NULL)
    {
        printf("WARNING: ADIOS2 initialization failed on process %d\n", my_rank);
        adios2_close(adios_engine);
        adios2_finalize(adios_object);
        return;
    }

    adios2_step_status status;

    adios2_begin_step(adios_engine, adios2_step_mode_read, -1, &status);

    // Get the global array variable already stored in file metadata
    adios2_variable *global_array_var = adios2_inquire_variable(adios_io, ADIOS_GLOBAL_ARRAY_VAR);

    if (global_array_var == NULL)
    {
        if (my_rank == RANK_ZERO)
            printf("WARNING: Could not find global array variable.\n");
    }

    else
    {
        // Even though we are reading back in an identical way to the write,
        // the metadata will not have saved this information. Therefore set it again.
        adios2_set_selection(global_array_var, DIMENSIONS, starts, size_t_local);

        // Read data into the 3D array according to structured already stored in metadata
        adios2_error error = adios2_get(adios_engine, global_array_var, &io_data[0][0][0], adios2_mode_deferred);

        adios2_end_step(adios_engine);

        if (error != adios2_error_none || status != adios2_step_status_ok)
        {
            if (my_rank == RANK_ZERO)
                printf("WARNING: ADIOS2 step did not complete successfully\n");
        }
    }

    adios2_close(adios_engine);
    adios2_finalize(adios_object);
}

/**
 * @brief Read data which has been written with ADIOS2 back and verify its correctness.
 *
 * Use a single process to read the ADIOS2 data which was written to file through the benchmark,
 * loading it into a 1D array and verifying that it forms the correct pattern (a series of incrementing
 * double-precision values starting from 1 and ending at global_sizes[0]*global_sizes[1]*global_sizes[2])
 *
 * @param[in] file_name Name of the ADIOS2 file (folder) to read from
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] communicator The global MPI communicator
 */
void adios2_verify(char *file_name, double *global_sizes, MPI_Comm communicator)
{
    int my_rank;

    MPI_Comm_rank(communicator, &my_rank);

    // Read back in serial for verification
    if (my_rank == RANK_ZERO)
    {
        adios2_adios *adios_object = adios2_init_config_mpi(ADIOS_CONFIG_FILE, MPI_COMM_SELF);
        adios2_io *adios_io = adios2_declare_io(adios_object, "adios_verify");
        adios2_engine *adios_engine = adios2_open(adios_io, file_name, adios2_mode_read);

        if (adios_object == NULL || adios_io == NULL || adios_engine == NULL)
        {
            printf("Debug mode failed to start ADIOS2\n");
            adios2_close(adios_engine);
            adios2_finalize(adios_object);
            return;
        }

        // Read into a 1D array same size as the flattened 3D array
        double *data_verify = malloc(global_sizes[0] * global_sizes[1] * global_sizes[2] * sizeof(double));

        if (data_verify == NULL)
            printf("Debug mode ADIOS2 read back malloc issue\n");

        adios2_step_status status;
        adios2_begin_step(adios_engine, adios2_step_mode_read, -1, &status);

        adios2_variable *read_data = adios2_inquire_variable(adios_io, ADIOS_GLOBAL_ARRAY_VAR);

        if (read_data == NULL)
            printf("Debug mode ADIOS2 issue finding global array variable\n");

        // Read data back into our 3D array to verify its correctness
        adios2_error error = adios2_get(adios_engine, read_data, &data_verify[0], adios2_mode_deferred);

        if (error != adios2_error_none || status != adios2_step_status_ok)
            printf("Debug mode ADIOS2 issue reading data back from file\n");

        adios2_end_step(adios_engine);
        adios2_close(adios_engine);

        adios2_finalize(adios_object);

        bool incorrect = false;
        double last_value = 0;

        int values_checked = 0;
        int values_to_check = global_sizes[0] * global_sizes[1] * global_sizes[2];
        double *current = &data_verify[0];

        // Go through the populated array to check it is correct
        while (values_checked < values_to_check)
        {
            if (*current == last_value + 1)
                last_value = *current;

            else
            {
                incorrect = true;
                break;
            }

            current++;
            values_checked++;
        }

        if (last_value != global_sizes[0] * global_sizes[1] * global_sizes[2])
            incorrect = true;

        if (incorrect)
        {
            // Rewind a bit to print prior values
            double *prior_print = current - DEBUG_PRINT_NUM_VALUES;

            // Rewound too far
            if (prior_print < &data_verify[0])
            {
                values_checked = 0;
                prior_print = &data_verify[0];
            }

            else
            {
                values_checked -= DEBUG_PRINT_NUM_VALUES;
                printf("... ");
            }

            last_value = *prior_print;
            printf("%lf ", last_value);
            prior_print++;

            while (values_checked < values_to_check)
            {
                printf("%lf ", *prior_print);

                if (*prior_print != last_value + 1)
                    break;

                last_value = *prior_print;
                prior_print++;

                values_checked++;
            }

            printf("<-- VALIDATION ERROR OCCURED HERE\n");
        }

        free(data_verify);
    }
}

/**
 * @brief Retreive the current ADIOS2 IO mode in use.
 *
 * Starts ADIOS2 from the configuration file and reads the IO engine to use
 *
 * @param[in] io_comm The global MPI communicator
 * @param[out] adios_io_mode Pointer to enum io_mode where to store the retreived IO mode
 *
 * @return Returns `true` if successfully read the IO mode of ADIOS2, `false` otherwise.
 */
bool get_adios2_io_mode(MPI_Comm io_comm, enum io_mode *adios_io_mode)
{
    int my_rank;
    MPI_Comm_rank(io_comm, &my_rank);

    adios2_adios *adios_object = adios2_init_config_mpi(ADIOS_CONFIG_FILE, io_comm);
    adios2_io *adios_io = adios2_declare_io(adios_object, ADIOS_IO_NAME);

    if (adios_object == NULL || adios_io == NULL)
    {
        if (my_rank == RANK_ZERO)
            printf("Failed to start ADIOS2 to retrieve io mode...\n");
        return false;
    }

    // Get information about what engine type is being used, e.g. HDF5, BP5...

    size_t engine_name_length;

    // First see how much space to malloc to store string
    adios2_engine_type(NULL, &engine_name_length, adios_io);

    // Set length plus null char
    char *adios2_type = malloc(engine_name_length + 1);
    adios2_type[engine_name_length] = NULLCHAR;

    if (adios2_type == NULL)
    {
        printf("WARNING: failed to malloc space for engine type on process %d\n", my_rank);
        return false;
    }

    bool found = true;

    adios2_engine_type(adios2_type, &engine_name_length, adios_io);

    // Check the IO mode string retreived against the known accepted strings

    if (equals_ignore_case(adios2_type, ADIOS_MODE_HDF5))
        *adios_io_mode = adios_hdf5;
    else if (equals_ignore_case(adios2_type, ADIOS_MODE_BP3))
        *adios_io_mode = adios_bp3;
    else if (equals_ignore_case(adios2_type, ADIOS_MODE_BP4))
        *adios_io_mode = adios_bp4;
    else if (equals_ignore_case(adios2_type, ADIOS_MODE_BP5))
        *adios_io_mode = adios_bp5;

    else
    {
        if (my_rank == RANK_ZERO)
            printf("WARNING: skipping ADIOS because unrecognized or unsupported ADIOS engine type %s\n", adios2_type);

        found = false;
    }

    // If succesful, report the identified engine information
    if (found && my_rank == RANK_ZERO)
        printf("Using ADIOS2 engine type: %s\n", adios2_type);

    free(adios2_type);

    adios2_finalize(adios_object);

    return found;
}

/**
 * @brief Clean up the files written by the native ADIOS modes.
 *
 * Remove the files written by ADIOS2 modes bp3, bp4, or bp5. Since ADIOS2 writes
 * a folder, with the files of the folder varying depending on which mode is used,
 * and the aggregator count, this removal uses 'brute force' to accomplish this; see details.
 *
 * @param[in] file_name The name of the file or resource to clean up.
 * @param[in] io_mode The ADIOS2 IO mode being used
 *
 * @return Returns `true` if cleanup was successful, `false` otherwise.
 *
 * @details
 * This is a bit of a workaround for deletion with ADIOS, which generates a set of files for its native binary formats.
 * The goal is to have a platform-independent solution to delete whatever file was generated.
 * Unfortunately not many good options exist, so solution here is to delete files one-by-one depending on mode used.
 * An additional complication is that many numbered files are sometimes generated, e.g. data.0, data.1, ...
 * This is solved here with a do-while loop, which just keeps incrementing a counter and deleting as long as it is successful.
 */
bool adios2_native_cleanup(char const *file_name, enum io_mode io_mode)
{
    // Handle Adios native binary format bp3
    if (io_mode == adios_bp3)
    {
        char adios_files[ADIOS_BP3_FILE_COUNT][MAX_FILENAME_LEN];

        for (int i = 0; i < ADIOS_BP3_FILE_COUNT; i++)
        {
            char temp[MAX_FILENAME_LEN];
            snprintf(temp, MAX_FILENAME_LEN, "%s%s", file_name, bp3_file_names[i]);

            int j = 0;

            // Remove the file and any related numbered file
            do
            {
                snprintf(adios_files[i], MAX_FILENAME_LEN, temp, j);
                j++;
            } while (remove(adios_files[i]) == 0);
        }

        char bp3_dir[MAX_FILENAME_LEN];
        snprintf(bp3_dir, MAX_FILENAME_LEN, "%s.bp.dir", file_name);

        // Finally try to delete the now hopefully empty sub-folder
        return remove(bp3_dir) == 0;
    }

    // Handle Adios native binary format bp4
    else if (io_mode == adios_bp4)
    {
        char adios_files[ADIOS_BP4_FILE_COUNT][MAX_FILENAME_LEN];

        for (int i = 0; i < ADIOS_BP4_FILE_COUNT; i++)
        {
            char temp[MAX_FILENAME_LEN];
            snprintf(temp, MAX_FILENAME_LEN, "%s%s", file_name, bp4_file_names[i]);

            int j = 0;

            // Remove the file and any related numbered file
            do
            {
                snprintf(adios_files[i], MAX_FILENAME_LEN, temp, j);
                j++;
            } while (remove(adios_files[i]) == 0);
        }
    }

    // Handle Adios native binary format bp5
    else if (io_mode == adios_bp5)
    {
        char adios_files[ADIOS_BP5_FILE_COUNT][MAX_FILENAME_LEN];

        for (int i = 0; i < ADIOS_BP5_FILE_COUNT; i++)
        {
            // Determine the file name
            char temp[MAX_FILENAME_LEN];
            snprintf(temp, MAX_FILENAME_LEN, "%s%s", file_name, bp5_file_names[i]);

            int j = 0;

            // Remove the file and any related numbered file
            do
            {
                snprintf(adios_files[i], MAX_FILENAME_LEN, temp, j);
                j++;
            } while (remove(adios_files[i]) == 0);
        }
    }

    return remove(file_name) == 0;
}

#endif

#ifdef NOADIOS

/**
 * @brief Dummy version of adios2_write, used when ADIOS2 is disabled.
 */
void adios2_write(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    return;
}

/**
 * @brief Dummy version of adios2_read, used when ADIOS2 is disabled.
 */
void adios2_read(char const *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm cartesian_comm)
{
    return;
}

/**
 * @brief Dummy version of adios2_verify, used when ADIOS2 is disabled.
 */
void adios2_verify(char *file_name, double *global_sizes, MPI_Comm communicator)
{
    return;
}
/**
 * @brief Dummy version of get_adios2_io_mode, used when ADIOS2 is disabled.
 *
 * Also print out a helpful warning to the user, since this method should not have
 * been called unless user asked for ADIOS2 but compiled without it.
 */
bool get_adios2_io_mode(MPI_Comm io_comm, enum io_mode *adios_io_mode)
{
    int my_rank;
    MPI_Comm_rank(io_comm, &my_rank);

    if (my_rank == RANK_ZERO)
        printf("Tried to use ADIOS2 io option, but benchio is currently compiled without ADIOS2.\n");

    return false;
}

/**
 * @brief Dummy version of adios2_native_cleanup, used when ADIOS2 is disabled.
 */
bool adios2_native_cleanup(char const *file_name, enum io_mode io_mode)
{
    return false;
}

#endif