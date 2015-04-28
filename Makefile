##################################
# Assuming you are using an Intel#
# compiler stack, change MPICC 	 #
# accordingly.					 #
##################################

OPTIMIZE = -O3 -funroll-all-loops
DEBUG = -g
CFLAGS = $(DEBUG) $(OPTIMIZE) -lrt

###############################################
# uncomment if you have BLAS installed		  #
# set the env. var MKLROOT before using 'make'#
###############################################
#BLAS = -DBLAS
#MKLLIBS = -L$(MKLROOT)/lib/intel64 -mkl -lmkl_intel_lp64

MPICC = mpiicpc

EXE = tdrqr 

all:    $(EXE)

tdrqr: main.cpp utils.cpp
	$(MPICC) $(CFLAGS) $(BLAS) $(MKLLIBS) -o $@ $^

clean:
	rm -f *.o *.log *.stdout errors* slurm-* $(EXE)
