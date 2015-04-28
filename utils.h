#ifndef UTILS_H
#define UTILS_H

// MPI
#include <mpi.h>

// C++ STL
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cmath>

// C libraries
#include <assert.h>
#include <sys/time.h>

#define DATE __DATE__
#define CVERSION __VERSION__

#define SOFTN "tdrqr"
#define SOFTV "Ver. 0.6"
#define BUILD "Built with MPI, no threading"

#define FLERR __FILE__,__LINE__
#define TAG_INIT        666
#define TAG_RESULT      1729
#define MASTER 0

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#ifdef BLAS
extern "C" {
	void dger_(int *M, int *N, double *ALPHA, double *X, int *INCX, double *Y, int *INCY, double *A, int *LDA);
}
#endif


/* Tri-diagonlization routines */
static void do_block (int lda, int nlocal, int M, int N, int K, double* A, double* B, double* C);
int matrixMultiply(double *a, double *b, double *c, int n, int n_local) ;
void house(double *H, double *A, int Adim, int iter);
void form_Q(int N, double *a, double *b, double *lambda, double **Q);
double secular(double bm, int n, double *d, double *xi, double x);
void bisection(double bm, int n, double *d, double *xi, double *lambda);

/* Matrix constructors */
void sym_matrix(int Adim, double *Asym);
double *identity_1d(int rows, int cols);

/* Error handling and memory management routines */
double *allocate_1d(int rows, int cols);
double *allocate_vector(int length);
double **allocate_2d(int rows, int cols);
int *allocate_int_1d(int rows, int cols);
int **allocate_int_2d(int rows, int cols);
void sdestroy_1d(double *matrix);
void sdestroy_2d(double **matrix, int width);
void kill(const char *file, int line, const char *message);

template <typename TYPE>
  TYPE *create(TYPE *&array, int rows, int cols)
  {
    array = (TYPE *)calloc(rows*cols,sizeof(TYPE)) ;
    return array;
  }

/*------------------------------------------------------------------------
Adjust slack of rows in case rowsTotal is not exactly divisible
--------------------------------------------------------------------------*/
inline int getRowCount(int rowsTotal, int iam, int world) {
    return (rowsTotal / world) + (rowsTotal % world > iam);
}

#endif

