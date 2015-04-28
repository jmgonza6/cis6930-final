# On Trestles we will check versus your performance versus Intel MKL library's BLAS.

#WARN = -Wall -Wextra -Wunused-variable -Wunused-parameter -Wunsafe-loop-optimizations
OPTIMIZE = -O3 -xSSE2
OPENMP = -openmp
DEBUG = -g
COFLAGS = $(DEBUG) $(WARN) $(OPTIMIZE) -lrt

SCALAPACK = -L$(MKLROOT)/lib/intel64 -mkl -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64

MKLLIBS = -L$(MKLROOT)/lib/intel64 -mkl -lmkl_intel_lp64

CC = icpc
MPICC = mpiicpc

TARGETS = trdqr 

all:    $(TARGETS)

trdqr: main-comments.cpp utils.cpp
	$(MPICC) $(COFLAGS) $(SCALAPACK) $(MKLLIBS) -o $@ $^

clean:
	rm -f *.o *.log *.stdout errors* slurm-* $(TARGETS)
