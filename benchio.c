/**
 * @file benchio.c
 * @brief Program starting point for benchio.
 *
 * Parse user input and determine the configuration to be used,
 * then run the program main benchmark loop accordingly.
 *
 * @section LICENSE
 * This software is released under the MIT License.
 */

#include <ctype.h>
#include <time.h>
#include <string.h>

#include "benchio.h"

/**
 * @brief The names of each of the io methods, in a lowercase format. Used for file creation etc.
 */
char const *const io_method_names[] = {"serial", "proc", "node", "mpiio", "hdf5", "netcdf", "adios"};

/**
 * @brief The names of each of the io methods, in a nice-to-print format. Used for user output.
 */
char const *const io_formatted_names[] = {"Serial", "Proc", "Node", "MPI-IO", "HDF5", "NetCDF", "Adios2"};

/**
 * @brief The names of each of the stripe methods, used for file creation etc.
 */
char const *const stripe_method_names[] = {"unstriped", "striped", "fullstriped"};

/**
 * @brief The legal arguments to accept. Used in process_args(...).
 *
 * See the struct definition in benchio.h for explanation of parameters.
 */
struct argument expected_arguments[] = {
    {"n1", "n1", false, false, false, false},    // n1: required
    {"n2", "n2", false, false, false, false},    // n2: required
    {"n3", "n3", false, false, false, false},    // n3: required
    {"scale", "sc", false, false, false, false}, // scale: required
    {"mode", "m", false, true, false, false},    // mode: optional
    {"stripe", "st", false, true, false, false}, // stripe: optional
    {"read", "r", true, true, false, false},     // read: optional and standalone
    {"help", "h", true, true, false, false}};    // help: optional and standalone

/**
 * @brief Main benchio starting point.
 *
 * Process arguments, identify the environment, and start the main loop of the benchmark.
 *
 * @param argc Number of arguments from user
 * @param argv Argument array from user
 * @return EXIT_SUCCESS on successful completion, otherwise EXIT_FAILURE
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int my_rank, world_size;

    // Identify initial MPI environment

    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm cartesian_comm;

    MPI_Comm_rank(global_comm, &my_rank);
    MPI_Comm_size(global_comm, &world_size);

    // Set up variables to store user input in

    bool use_io_method[IO_METHOD_COUNT] = {false, false, false};
    bool use_stripe_method[STRIPE_TYPE_COUNT] = {false, false, false};

    bool read_benchmark = false;

    double local_sizes[DIMENSIONS] = {0, 0, 0};
    int input_sizes[DIMENSIONS] = {0, 0, 0};

    bool global_flag = false;

    double ***io_data;

    // Used to indicate if user ran benchio with --help or -h flags
    bool user_wants_help = false;

    // Fill user input from arguments
    bool ok_to_proceed = process_args(argc, argv, my_rank, input_sizes, &global_flag, use_io_method, use_stripe_method, &read_benchmark, &user_wants_help);

    // Cannot continue either because user asked for help or because args were incorrect
    if (!ok_to_proceed)
    {
        // Incorrect args
        if (!user_wants_help)
        {
            if (my_rank == RANK_ZERO)
                print_simple_usage();

            MPI_Finalize();
            return EXIT_FAILURE;
        }

        // --help or -h flag
        else
        {
            print_detailed_usage();
            MPI_Finalize();
            return EXIT_SUCCESS;
        }
    }


    // Init to 0 because any non zero value indicates to MPI a requirement of that number of procs
    int process_dims[DIMENSIONS] = {0, 0, 0};

    // Let MPI determine a good process distribution for a cartesian topology
    MPI_Dims_create(world_size, DIMENSIONS, process_dims);

    if (global_flag)
    {
        // Strong scaling with "global" flag. Each process gets subset of user input totals.
        local_sizes[0] = input_sizes[0] / process_dims[0];
        local_sizes[1] = input_sizes[1] / process_dims[1];
        local_sizes[2] = input_sizes[2] / process_dims[2];
    }
    else
    {
        // Weak scaling. Each process has specified amount of data
        local_sizes[0] = input_sizes[0];
        local_sizes[1] = input_sizes[1];
        local_sizes[2] = input_sizes[2];
    }

    double global_sizes[DIMENSIONS];
    global_sizes[0] = local_sizes[0] * process_dims[0];
    global_sizes[1] = local_sizes[1] * process_dims[1];
    global_sizes[2] = local_sizes[2] * process_dims[2];

    // If using global scaling, this additional check verifies that the user got the global size requested. If not, suggest nearest split and abort.
    if (global_flag && (global_sizes[0] != input_sizes[0] || global_sizes[1] != input_sizes[1] || global_sizes[2] != input_sizes[2]))
    {
        if (my_rank == RANK_ZERO)
            printf("Failed to divide processes evenly across dimensions. Suggested global size: (%.0lf, %.0lf, %.0lf)\n", global_sizes[0], global_sizes[1], global_sizes[2]);

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // We do not want any periodicity in our process grid
    int const periodic[DIMENSIONS] = {false, false, false};

    bool const reorder = false;

    // Create a cartesian topology
    MPI_Cart_create(global_comm, DIMENSIONS, process_dims, periodic, reorder, &cartesian_comm);

    if (my_rank == RANK_ZERO)
    {
        printf("\nSimple Parallel IO benchmark\n");
        printf(SEPARATOR_STRING);
        char const *process_processes = world_size == 1 ? "process" : "processes";
        printf("Running on %d %s\n", world_size, process_processes);

        if (DEBUG_MODE)
            printf("DEBUG MODE IS ENABLED\n");
    }

    MPI_Comm node_comm;
    int node_num;

    // Determine information about how many nodes were are running on,
    // the names of the nodes, and process ranks within those nodes.
    // Used to print out information and for node/proc modes
    setup_nodes(cartesian_comm, my_rank, &node_comm, &node_num);

    double global_size_gib = 0;

    // Print out information relating to setup and process decomposition
    if (my_rank == RANK_ZERO)
    {
        int double_size;
        MPI_Type_size(MPI_DOUBLE_PRECISION, &double_size);

        double global_size_elements = global_sizes[0] * global_sizes[1] * global_sizes[2];
        global_size_gib = (global_size_elements * (double)double_size) / GIB;

        if (DEBUG_MODE && global_size_gib > 1.0)
            printf("WARNING: writing %lf GiB data in debug mode. May be slow.\n", global_size_gib);

        printf("\nProcess grid is (%d, %d, %d)\n", process_dims[0], process_dims[1], process_dims[2]);
        printf("Array size is (%.0lf, %.0lf, %.0lf)\n", local_sizes[0], local_sizes[1], local_sizes[2]);
        printf("Global size is (%.0lf, %.0lf, %.0lf)\n", global_sizes[0], global_sizes[1], global_sizes[2]);
        printf("\nTotal amount of data = %lf GiB\n", global_size_gib);
        printf("\nClock resolution is %f ns\n", MPI_Wtick() * 1E9);
        printf("\nUsing the following IO methods\n");
        printf(SEPARATOR_STRING);

        for (int i = 0; i < IO_METHOD_COUNT; i++)
        {
            if (use_io_method[i])
                printf("%s\n", io_method_names[i]);
        }

        printf("\nUsing the following stripings\n");
        printf(SEPARATOR_STRING);

        for (int i = 0; i < STRIPE_TYPE_COUNT; i++)
        {
            if (use_stripe_method[i])
                printf("%s\n", stripe_method_names[i]);
        }

        printf("\n");
    }

    // Dynamically allocate a 3D array on each process
    io_data = (double ***)arraymalloc3d(local_sizes[0], local_sizes[1], local_sizes[2], sizeof(double));

    if (io_data == NULL)
    {
        printf("malloc failed on process %d. Aborting.\n", my_rank);
        MPI_Abort(global_comm, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int coords[DIMENSIONS];
    MPI_Cart_coords(cartesian_comm, my_rank, DIMENSIONS, coords);

    // This is the step where each process's array is filled with data
    populate_io_data(local_sizes, coords, global_sizes, io_data);

    enum io_mode adios_io_mode = none;

    if (use_io_method[ADIOS2_IO_IDX])
    {
        // If benchio is compiled without adios2 (NOADIOS flag), then this will return false and print out a warning
        bool success = get_adios2_io_mode(cartesian_comm, &adios_io_mode);

        if (!success)
            use_io_method[ADIOS2_IO_IDX] = false;

        else if (adios_io_mode == adios_hdf5)
        {
            if (my_rank == RANK_ZERO)
                printf("ADIOS2 HDF5 option currently unsupported in benchio, so ADIOS2 was disabled.\n");

            use_io_method[ADIOS2_IO_IDX] = false;
        }
    }

    // Run the benchmark using the specified/identified configuration
    main_benchmark_loop(cartesian_comm, node_comm, use_stripe_method, use_io_method, local_sizes, global_sizes, global_size_gib,
                        read_benchmark, node_num, adios_io_mode, coords, io_data);

    if (my_rank == RANK_ZERO)
    {
        printf(SEPARATOR_STRING);
        printf("Finished\n");
        printf(SEPARATOR_STRING);
    }

    free(io_data);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/**
 * @brief Start the benchmark, looping through each step according to configuration, and report results.
 *
 * Take some identified set of settings, such as the IO methods to use,
 * the stripings to use, and whether or not to run a read-back benchmark,
 * and run the benchmark accordingly.
 *
 * @param[in] cartesian_comm An MPI communicator with a cartesian topology
 * @param[in] node_comm An MPI communicator specific to the current process' node
 * @param[in] use_stripe_method An array of booleans indicating whether or not to use each striping method
 * @param[in] use_io_method An array of booleans indicating whether or not to use each io method
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] global_size_gib The total global array size in GiB.
 * @param[in] read_benchmark Boolean flag indicating whether or not to also perform a read benchmark
 * @param[in] node_num The rank of the process' current node in the node-spanning communicator
 * @param[in] adios_io_mode Which underlying IO mode ADIOS is using, if applicable
 * @param[in] coords The coordinates of the current process in cartesian_comm
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 */
void main_benchmark_loop(MPI_Comm cartesian_comm, MPI_Comm node_comm, bool *use_stripe_method, bool *use_io_method, double *local_sizes,
                         double *global_sizes, double global_size_gib, bool read_benchmark, int node_num, enum io_mode adios_io_mode,
                         int *coords, double ***io_data)
{
    int my_rank;
    MPI_Comm_rank(cartesian_comm, &my_rank);

    // Loop through each of the IO methods available
    for (int io = 0; io < IO_METHOD_COUNT; io++)
    {

        // Skip IO method if it is not enabled
        if (!use_io_method[io])
            continue;

        enum io_mode io_mode = none;

        // Print out information about current IO method being used
        if (my_rank == RANK_ZERO)
        {
            printf(SEPARATOR_STRING);
            printf("%s\n", io_formatted_names[io]);
            printf(SEPARATOR_STRING);
        }

        // Loop through each of the striping methods available
        for (int st = 0; st < STRIPE_TYPE_COUNT; st++)
        {
            // Skip striping method if it is not enabled
            if (!use_stripe_method[st])
                continue;

            MPI_Comm io_comm = cartesian_comm;

            // Determine file names depending on mode in use

            char file_name[MAX_FILENAME_LEN];
            char suffix[MAX_SUFFIX_LEN] = "";

            if (io == PROC_IO_IDX)
            {
                io_comm = MPI_COMM_SELF;
                snprintf(suffix, MAX_SUFFIX_LEN, "%06d", my_rank);
            }

            else if (io == NODE_IO_IDX)
            {
                io_comm = node_comm;
                snprintf(suffix, MAX_SUFFIX_LEN, "%06d", node_num);
            }

            snprintf(file_name, MAX_FILENAME_LEN, "%s/%s%s.dat", stripe_method_names[st], io_method_names[io], suffix);

            if (my_rank == RANK_ZERO)
            {
                printf("File writing to / reading from: %s\n", file_name);
            }

            // Set io mode information (mostly important for ADIOS2 since it can have different underlying modes)

            if (io == SERIAL_IO_IDX || io == NODE_IO_IDX || io == PROC_IO_IDX)
                io_mode = serial;

            else if (io == MPIIO_IO_IDX)
                io_mode = mpiio;

            else if (io == ADIOS2_IO_IDX)
                io_mode = adios_io_mode;

            // Run the writing-based benchmark
            bool success = run_write_benchmark(file_name, io_data, local_sizes, global_sizes, io_comm, my_rank, io, global_size_gib, io_mode);

            if (!success)
                break;

            // Run the reading-based benchmark
            if (read_benchmark)
            {
                bool success = run_read_benchmark(file_name, local_sizes, global_sizes, io_comm, my_rank, io, global_size_gib, io_mode, cartesian_comm, coords);

                if (!success)
                    break;
            }

            // File deletion. Process 0 of io_comm deletes file
            if (!boss_delete(io_mode, file_name, io_comm))
                printf("Failed to delete generated file on rank %d\n", my_rank);
        }
    }
}

/**
 * @brief Run the read-based benchmark according to specified configuration.
 *
 * Run the read-based benchmark and report results. If DEBUG_MODE is enabled, this function
 * also validates the data which was read back in for correctness.
 *
 * @param[in] file_name Name of the file to read from
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] io_comm The MPI communicator context to perform the read in
 * @param[in] my_rank The rank of the current process in the global communicator
 * @param[in] io An integer indicating which IO mode to read using; see benchio.h compile-time constants
 * @param[in] global_size_gib The size of the global array in GiB.
 * @param[in] io_mode Which underlying IO mode is in use (ADIOS2 has multiple engines)
 * @param[in] cartesian_comm The cartesian-topology communicator which was used to populate data
 * @param[in] coords Coordinates of current process in cartesian_comm, size DIMENSIONS
 *
 * @return Returns `true` if benchmark completed normally, `false` otherwise.
 *
 */
bool run_read_benchmark(char *file_name, double *local_sizes, double *global_sizes, MPI_Comm io_comm, int my_rank, int io, double global_size_gib, enum io_mode io_mode, MPI_Comm cartesian_comm, int *coords)
{
    // Dynamically allocate a 3D array to read into on each process
    // Use a separate buffer here just to be on the safe side and not overwrite original buffer
    double ***read_io_data = (double ***)arraymalloc3d(local_sizes[0], local_sizes[1], local_sizes[2], sizeof(double));

    if (read_io_data == NULL)
    {
        printf("malloc failed on process %d. Aborting.\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    // Ensure that all processes start the benchmark at the same time
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    if (io == SERIAL_IO_IDX || io == NODE_IO_IDX || io == PROC_IO_IDX)
        serial_read(file_name, read_io_data, local_sizes, io_comm);

    else if (io == MPIIO_IO_IDX)
        mpiio_read(file_name, read_io_data, local_sizes, global_sizes, io_comm);

    else if (io == ADIOS2_IO_IDX)
        adios2_read(file_name, read_io_data, local_sizes, global_sizes, io_comm);

    else
    {
        if (my_rank == RANK_ZERO)
            printf("Could not perform read for IO method %s because it is currently unsupported.\n", io_method_names[io]);
        return false;
    }

    // Ensure that all processes have finished before recording end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Print out results
    if (my_rank == RANK_ZERO)
    {
        double time_taken = end_time - start_time;
        double io_rate = global_size_gib / time_taken;

        printf("read time = %fs, rate = %f GiB/s\n", time_taken, io_rate);
    }

    // Verify correctness, if applicable
    if (DEBUG_MODE && io_mode != serial)
    {
        if (my_rank == RANK_ZERO)
            printf("Input verification started...\n");

        verify_input(cartesian_comm, read_io_data, coords, local_sizes, global_sizes);

        if (my_rank == RANK_ZERO)
            printf("Input verification finished.\n");
    }

    free(read_io_data);

    return true;
}

/**
 * @brief Run the write-based benchmark according to specified configuration.
 *
 * Run the write-based benchmark and report results. If DEBUG_MODE is enabled, this function
 * also validates the data which was written by reading it back in and checking it for correctness.
 *
 * @param[in] file_name Name of the file to read from
 * @param[in] io_data The current process' 3D array of data forming its part of the global array
 * @param[in] local_sizes An array of values indicating the per-process array dimensions
 * @param[in] global_sizes An array of values indicating the global array dimensions
 * @param[in] io_comm The MPI communicator context to perform the write in
 * @param[in] my_rank The rank of the current process in the global communicator
 * @param[in] io An integer indicating which IO mode to read using; see benchio.h compile-time constants
 * @param[in] global_size_gib The size of the global array in GiB.
 * @param[in] io_mode Which underlying IO mode is in use (ADIOS2 has multiple engines)
 *
 * @return Returns `true` if benchmark completed normally, `false` otherwise.
 *
 */
bool run_write_benchmark(char *file_name, double ***io_data, double *local_sizes, double *global_sizes, MPI_Comm io_comm, int my_rank, int io, double global_size_gib, enum io_mode io_mode)
{
    // Ensure that all processes start the benchmark at the same time
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    if (io == SERIAL_IO_IDX || io == NODE_IO_IDX || io == PROC_IO_IDX)
        serial_write(file_name, io_data, local_sizes, io_comm);

    else if (io == MPIIO_IO_IDX)
        mpiio_write(file_name, io_data, local_sizes, global_sizes, io_comm);

    else if (io == ADIOS2_IO_IDX)
        adios2_write(file_name, io_data, local_sizes, global_sizes, io_comm);

    else
    {
        if (my_rank == RANK_ZERO)
            printf("Could not perform IO method %s because it is currently unsupported.\n", io_method_names[io]);
        return false;
    }

    // Ensure that all processes have finished before recording end time
    MPI_Barrier(MPI_COMM_WORLD);

    // Print out results
    if (my_rank == RANK_ZERO)
    {
        double end_time = MPI_Wtime();

        double time_taken = end_time - start_time;
        double io_rate = global_size_gib / time_taken;

        printf("write time = %fs, rate = %f GiB/s\n", time_taken, io_rate);
    }

    // Verify correctness, if applicable
    if (DEBUG_MODE && io_mode != serial)
    {
        if (my_rank == RANK_ZERO)
            printf("Output verification started...\n");

        verify_output(io, file_name, global_sizes, io_comm);

        if (my_rank == RANK_ZERO)
            printf("Output verification finished.\n");
    }

    return true;
}

/**
 * @brief Fill the current process' array with data consistent with the global array.
 *
 * Fill the current process' array with data such that, in the cartesian grid of
 * processes, it forms part of a series of incrementing double-precision values from
 * 1 to global_sizes[0]*global_sizes[1]*global_sizes[2].
 *
 * @param[in] local_sizes The array of sizes on the current process
 * @param[in] coords The array of coordinates in the cartesian communuicator
 * @param[in] global_sizes The array of sizes of the global array
 * @param[in, out] io_data Pointer to a pre-allocated 3D array to populate
 *
 * @details
 * Populate the array of the current process.
 * Each element contains a globally unique double-precision value between 1 and global_sizes[0]*global_sizes[1]*global_sizes[2].
 * Count increments by one for each element.
 *
 * Example ordering on a (2, 2, 1) decomposition with 2x2x2 local data.
 * Each cube is a process' own array. P0 is bottom-left process.
 * With more processes, extend pattern...
 *
 *  ```
 *        20          24       28         32
 *         +----------+        +----------+
 *        /|         /|       /|         /|
 *       / |       /  |      / |       /  |
 *    4 +--------+ 8  |  12 +--------+ 16 |
 *      |  |     |    |     |  |     |    |
 *      |  +----------+     |  +----------+
 *      | / 19   | 23/      | / 27   | 31/
 *      |/       | /        |/       | /
 *    3 +--------+ 7     11 +--------+ 15
 *         18         22        26         30
 *         +----------+        +----------+
 *        /|         /|       /|         /|
 *       / |       /  |      / |       /  |
 *    2 +--------+ 6  |  10 +--------+ 14 |
 *      |  |     |    |     |  |     |    |
 *      |  +----------+     |  +----------+
 *      | / 17   | 21/      | / 25   | 29/
 *      |/       | /        |/       | /
 *    1 +--------+ 5      9 +--------+ 13
 *
 *
 *       k      j
 *       |     /
 *       |    /
 *       |   /
 *       |  /
 *       | /
 *       |/
 *       +------------ i
 *     array_example[j][i][k]
 *
 *     Notice that MPI structures its process grid as follows:
 *     mpi_dims_example[i][k][j] 
 * ```
 *  
 */
void populate_io_data(double *local_sizes, int *coords, double *global_sizes, double ***io_data)
{
    // Loop through each of the coordinates of the local array

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
                io_data[j_local][i_local][k_local] = k_global + i_global * global_sizes[2] + j_global * global_sizes[1] * global_sizes[2] + 1.0;
            }
        }
    }
}

/**
 * @brief Parse command-line arguments provided by user.
 *
 * The the command-line arguments provided by the user and validate them.
 * Print useful debug to user and return if some of their input was invalid, otherwise
 * quietly fill the parameters with the specified input.
 *
 * @param[in] argc Number of arguments from user
 * @param[in] argv Argument array from user
 * @param[in] my_rank Current process rank in the global MPI communicator
 * @param[out] sizes Array of size DIMENSIONS to store the user input for dimensions
 * @param[out] global_flag Pointer to boolean to store flag of whether to use local or global scaling
 * @param[out] use_io_method Array of size IO_METHOD_COUNT to store boolean values indicating whether to use each IO method
 * @param[out] use_stripe_method Array of size STRIPE_TYPE_COUNT to store boolean values indicating whether to use each striping method
 * @param[out] read_benchmark Pointer to a boolean to store flag of whether to also perform read-based benchmark
 * @param[out] user_wants_help Pointer to a boolean to store flag of whether user asked for help through --help or -h flags
 *
 * @return Return `true` if OK to continue benchmark, otherwise `false`
 */
bool process_args(int argc, char **argv, int my_rank, int *sizes, bool *global_flag, bool *use_io_method, bool *use_stripe_method, bool *read_benchmark, bool *user_wants_help)
{
    int arguments_to_look_for = sizeof(expected_arguments) / sizeof(expected_arguments[0]);

    struct argument *current_argument = NULL;

    // Used to indicate whether the last flag that was passed is ready to move on from
    bool ready_for_next = true;

    // The name of the last flag, used for printing out user feedback
    char *last_flag = "";

    // Flags to indicate whether or not any io / stripe methods were provided;
    // if not, then all are enabled
    bool found_any_io = false;
    bool found_any_stripe = false;

    // Loop through all the arguments and attempt to process them
    for (int i = 1; i < argc; i++)
    {
        bool short_arg = false;
        bool long_arg = false;
        char *argument_no_prefix;

        // Check if used long argument prefix (--)
        if (strncmp(argv[i], LONG_ARG, sizeof(LONG_ARG) - 1) == 0)
        {
            long_arg = true;
            argument_no_prefix = argv[i] + (sizeof(LONG_ARG) - 1);
        }

        // Check if used short argument prefix (-)
        // This needs to be checked AFTER we check for long prefix
        else if (strncmp(argv[i], SHORT_ARG, sizeof(SHORT_ARG) - 1) == 0)
        {
            short_arg = true;
            argument_no_prefix = argv[i] + (sizeof(SHORT_ARG) - 1);
        }

        // A flag was passed, but last flag does not have enough input
        if ((short_arg || long_arg) && !ready_for_next)
        {
            if (my_rank == RANK_ZERO)
                printf("Too few parameters passed to %s\n", last_flag);
            return false;
        }

        // A flag was passed and OK to process next one
        else if ((short_arg || long_arg) && ready_for_next)
        {
            bool match = false;

            // Try to find the argument in list of expected args
            for (int j = 0; j < arguments_to_look_for; j++)
            {
                char *comparison;

                if (short_arg)
                    comparison = expected_arguments[j].short_arg;
                else
                    comparison = expected_arguments[j].long_arg;

                if (equals_ignore_case(comparison, argument_no_prefix))
                {
                    // Found a match for the provided flag

                    match = true;
                    last_flag = argv[i];
                    current_argument = &expected_arguments[j];
                    expected_arguments[j].found = true;

                    // Arguments that do not need any additional info
                    if (expected_arguments[j].standalone)
                    {
                        if (equals_ignore_case(expected_arguments[j].long_arg, "read"))
                        {
                            *read_benchmark = true;
                            current_argument->complete = true;
                        }
                        else if (equals_ignore_case(expected_arguments[j].long_arg, "help"))
                        {
                            *user_wants_help = true;
                            return false;
                        }

                        ready_for_next = true;
                    }

                    else
                        ready_for_next = false;

                    // Found match, no need to keep iterating
                    break;
                }
            }

            // Could not match flag with any known flags
            if (!match)
            {
                if (my_rank == RANK_ZERO)
                    printf("Invalid flag \"%s\" provided.\n", argv[i]);
                return false;
            }
        }

        // Not processing a new flag
        else if (!(short_arg || long_arg))
        {
            // Was expecting a flag
            if (current_argument == NULL)
            {
                if (my_rank == RANK_ZERO)
                    printf("Invalid syntax \"%s\"\n", argv[i]);
                return false;
            }

            // Current flag is marked as complete, but more information was passed to it
            if (current_argument->complete)
            {
                if (my_rank == RANK_ZERO)
                    printf("Too many parameters passed to flag \"%s\"\n", last_flag);
                return false;
            }

            // Take action depending on current flag

            // n1, n2, n3 flags
            if (equals_ignore_case(current_argument->long_arg, "n1") || equals_ignore_case(current_argument->long_arg, "n2") || equals_ignore_case(current_argument->long_arg, "n3"))
            {
                int the_number;

                // Convert to number assuming input correct
                bool parse_success = string_to_integer(argv[i], &the_number);

                if (parse_success)
                {
                    if (equals_ignore_case(current_argument->long_arg, "n1"))
                        sizes[0] = the_number;
                    else if (equals_ignore_case(current_argument->long_arg, "n2"))
                        sizes[1] = the_number;
                    else if (equals_ignore_case(current_argument->long_arg, "n3"))
                        sizes[2] = the_number;

                    // Mark complete because we do not want more input for n1
                    current_argument->complete = true;
                }

                else
                {
                    if (my_rank == RANK_ZERO)
                        printf("Invalid argument \"%s\" for %s. Must be positive integer.\n", argv[i], current_argument->long_arg);
                    return false;
                }
            }

            // Strong or weak scaling flag
            else if (equals_ignore_case(current_argument->long_arg, "scale"))
            {
                if (equals_ignore_case(argv[i], "local"))
                    *global_flag = false;

                else if (equals_ignore_case(argv[i], "global"))
                    *global_flag = true;

                else
                {
                    if (my_rank == RANK_ZERO)
                        printf("Invalid argument \"%s\" for scale. Must be either \"local\" or \"global\".\n", argv[i]);

                    return false;
                }
            }

            // IO modes
            else if (equals_ignore_case(current_argument->long_arg, "mode"))
            {
                bool found_arg = false;

                for (int j = 0; j < IO_METHOD_COUNT; j++)
                {
                    if (equals_ignore_case(argv[i], io_method_names[j]))
                    {
                        use_io_method[j] = true;
                        found_any_io = true;
                        found_arg = true;
                        break;
                    }
                }

                if (!found_arg)
                {
                    if (my_rank == RANK_ZERO)
                        printf("Invalid IO method \"%s\".\n", argv[i]);

                    return false;
                }
            }

            // Unstriped, striped, or fullstriped
            else if (equals_ignore_case(current_argument->long_arg, "stripe"))
            {
                bool found_arg = false;

                for (int j = 0; j < STRIPE_TYPE_COUNT; j++)
                {
                    if (equals_ignore_case(argv[i], stripe_method_names[j]))
                    {
                        use_stripe_method[j] = true;
                        found_arg = true;
                        found_any_stripe = true;
                        break;
                    }
                }

                if (!found_arg)
                {
                    if (my_rank == RANK_ZERO)
                        printf("Invalid stripe argument \"%s\".\n", argv[i]);

                    return false;
                }
            }

            // This should not happen unless there is a bug, because the flag should be valid at this point
            else
            {
                if (my_rank == RANK_ZERO)
                    printf("An error occurred processing flag \"%s\" and argument \"%s\".\n", current_argument->long_arg, argv[i]);

                return false;
            }

            // Processed the current argument
            ready_for_next = true;
        }
    }

    // Verify that all required flags were found
    for (int i = 0; i < arguments_to_look_for; i++)
    {
        if (!expected_arguments[i].optional && !expected_arguments[i].found)
        {
            if (my_rank == RANK_ZERO)
                printf("Could not find required flag \"-%s\".\n", expected_arguments[i].short_arg);
            return false;
        }
    }

    // No optional IO arguments provided so enable all IO methods
    if (!found_any_io)
    {
        for (int i = 0; i < IO_METHOD_COUNT; i++)
            use_io_method[i] = true;

// Should not enable adios io mode if compiled without it
#ifdef NOADIOS
        use_io_method[ADIOS2_IO_IDX] = false;
#endif
    }

    // No optional striping arguments provided so enable all striping methods
    if (!found_any_stripe)
    {
        for (int i = 0; i < STRIPE_TYPE_COUNT; i++)
            use_stripe_method[i] = true;
    }

    return true;
}

/**
 * @brief Set up communicators reflecting the nodal environment currently in use.
 *
 * Create node-spanning communicators and identify the node bosses (rank 0s) of each
 * node. Also identify and print out the name of each node.
 *
 * @param[in] communicator An MPI communicator with all processes in it
 * @param[in] original_rank Rank of current process within `communicator`
 * @param[out] node_comm Pointer to communicator to fill with a communicator spanning the current node
 * @param[out] node_number Pointer to integer which will be filled with the node number of the current process
 */
void setup_nodes(MPI_Comm communicator, int original_rank, MPI_Comm *node_comm, int *node_number)
{
    int node_size, node_rank;
    int number_of_nodes;

    int node_name_len;
    char node_name[MPI_MAX_PROCESSOR_NAME];

    // Create a separate communicator for each node
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, original_rank, MPI_INFO_NULL, node_comm);

    MPI_Comm_size(*node_comm, &node_size);
    MPI_Comm_rank(*node_comm, &node_rank);

    // Two colors: one for rank-0 procs in the node communicator, one for all others
    int color = node_rank == RANK_ZERO ? node_rank : 1;

    // Rank-0 procs of each node form a single communicator, the rest get a single, junk, comm
    MPI_Comm node_boss_comm;

    MPI_Comm_split(communicator, color, original_rank, &node_boss_comm);

    // Rank-0 procs of each node now hold correct data, but not the others
    MPI_Comm_size(node_boss_comm, &number_of_nodes);
    MPI_Comm_rank(node_boss_comm, node_number);

    // Use a broadcast within each node eminating from each node communicator's rank 0 to let other procs know of correct node number
    MPI_Bcast(node_number, 1, MPI_INT, RANK_ZERO, *node_comm);

    if (original_rank == RANK_ZERO)
    {
        char const *node_nodes = number_of_nodes == 1 ? "node" : "nodes";

        printf("Running on %d %s\n", number_of_nodes, node_nodes);

        for (int i = 0; i < number_of_nodes; i++)
        {
            // Receive from all other node bosses
            if (i > 0)
            {
                MPI_Status recv_status;
                MPI_Recv(node_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, MPI_ANY_TAG, node_boss_comm, &recv_status);
                node_size = recv_status.MPI_TAG;
            }
            // In own process' case, we can just query the name directly
            else
            {
                MPI_Get_processor_name(node_name, &node_name_len);
            }

            char const *process_processes = node_size == 1 ? "process" : "processes";
            printf("Node number %d is %s with %d %s\n", i, node_name, node_size, process_processes);
        }
    }
    else if (node_rank == RANK_ZERO)
    {
        // Send name of node, along with size through the tag

        // Get the name of each node
        MPI_Get_processor_name(node_name, &node_name_len);

        int tag = node_size;
        MPI_Ssend(node_name, node_name_len, MPI_CHAR, RANK_ZERO, tag, node_boss_comm);
    }
}