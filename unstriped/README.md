This is the directory for testing IO rates when the file is not
striped across OSTs, i.e. the file system uses a single write
stream.

This directory should be set to have a single stripe.
The Lustre command to do this is: lfs setstripe -c 1 unstriped