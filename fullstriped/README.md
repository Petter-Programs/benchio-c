This is the directory for testing IO rates when the file is striped
across many OSTs, i.e. the file system uses multiple write streams.

This directory should be set to have the maximum number of stripes.
The Lustre command to do this is: lfs setstripe -c -1 fullstriped