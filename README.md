# high_performance_computing

Remember to switch compilers from the default (module load gcc)

Right now I started using a native way of implementing the matrices (double pointers) in C language, so our makefile is Makefile.gcc. We might want to use suncc later, but it would mean changes in "Makefile.suncc" and in the way of calling dgemm

Command to make our library: "make -f Makefile.gcc"
Command to call run the shell script: "bsub < mm_batch.sh"

There is also collect_batch file that as far as I understand can be used to run the calculations in a loop.

README file provided by the teachers explains how to modify the shell script in order to run specific functions with specific settings.
