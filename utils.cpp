#include "utils.h"

#ifdef BLAS
void blas_dger (int N, double* X, double* Y, double* A)
{
    int M = N;
    int LDA = N;
    double ALPHA = 1.;
    int INCX = 1;
    int INCY = 1;
    dger_(&M, &N, &ALPHA, X, &INCX, Y, &INCY, A, &LDA);
}
#endif



/*------------------------------------------------------------------------
    Form the Householder transformation matrix, Omega
--------------------------------------------------------------------------*/
void house(double *H, double *A, int Adim, int iter)
{
    #ifdef BLAS
    double *v = allocate_1d(Adim,Adim);
    #else
    double v[Adim][1], vt[1][Adim];
    #endif

    double *eye = identity_1d(Adim,Adim);
    double *W = allocate_1d(Adim,Adim);
  
    int ii, ll;

    double alpha =0.;
    for(ii=iter+1; ii<Adim; ii++)
        alpha += A[iter+ii*Adim]*A[iter+ii*Adim];

    alpha = A[iter+(iter+1)*Adim] >= 0. ? -sqrt(alpha) : sqrt(alpha);

    double rsq = sqrt(0.5*(alpha*alpha - alpha*A[iter+(iter+1)*Adim]));

    #ifdef BLAS
    for(ii=0; ii<Adim; ii++) {
        if(ii<=iter) {
            v[ii] = 0.;
        } else if (ii==iter+1) {
            v[ii] = (A[ii+iter*Adim] - alpha)/(2.*rsq);
        } else {
            v[ii] = (A[ii+iter*Adim])/(2.*rsq);
        }
    }
    #else
    for(int ii=0; ii<Adim; ii++) {
        if(ii<=iter) {
            v[ii][0] = 0.;
            vt[0][ii] = 0.;
        } else if (ii==iter+1) {
            v[ii][0] = (A[ii+iter*Adim] - alpha)/(2.*rsq);
            vt[0][ii] = v[ii][0];
        } else {
            v[ii][0] = (A[ii+iter*Adim])/(2.*rsq);
            vt[0][ii] = v[ii][0];
        }
    }
    #endif

    /* Outer product of v*v^T */
    int length = Adim*Adim;
    #ifdef BLAS
    blas_dger (Adim, v, v, W);

    for(ll=0; ll<(length - 3); ll+=4) {
        *(W + ll + 0) = *(W + ll + 0)*2.;
        *(W + ll + 1) = *(W + ll + 1)*2.;
        *(W + ll + 2) = *(W + ll + 2)*2.;
        *(W + ll + 3) = *(W + ll + 3)*2.;   
    }
    for(;ll<(length);ll++) {
        *(W + ll + 0) = *(W + ll + 0)*2.;
    }
    #else
    for(int ii=0; ii<Adim; ii++) {
        for(int jj=0; jj<Adim; jj++) {
            W[ii+jj*Adim] = 2.*v[ii][0]*vt[0][jj];
        }
    }
    #endif
    

    /* Construct Householder matrix */
    for(ii=0; ii<Adim; ii++) {
        for(ll=0; ll<(Adim-3); ll+=4) {
            *(H + ii + (ll+0)*Adim) = *(eye + ii + (ll+0)*Adim) - *(W + ii + (ll+0)*Adim);
            *(H + ii + (ll+1)*Adim) = *(eye + ii + (ll+1)*Adim) - *(W + ii + (ll+1)*Adim);
            *(H + ii + (ll+2)*Adim) = *(eye + ii + (ll+2)*Adim) - *(W + ii + (ll+2)*Adim);
            *(H + ii + (ll+3)*Adim) = *(eye + ii + (ll+3)*Adim) - *(W + ii + (ll+3)*Adim);
        }
        for(;ll<Adim;ll++) {
            *(H + ii + (ll+0)*Adim) = *(eye + ii + (ll+0)*Adim) - *(W + ii + (ll+0)*Adim);
        }
    }
    sdestroy_1d(W); sdestroy_1d(eye); 
    #ifdef BLAS
    sdestroy_1d(v);
    #endif
}
void form_Q(int N,  double *a, double *b, double *lambda, double **Q)
{
    if(N == 1){
        Q[0][0] = 1.0;
        lambda[0] = a[0];
    } else{
        int i,j,k,N1,N2,cnt;
        N1 = N/2;
        N2 = N-N1;

        double *d = allocate_vector(N);
        double *xi = allocate_vector(N);

        double **Q1 = allocate_2d(N1,N1);
        double **Q2 = allocate_2d(N2,N2);

        a[N1-1] = a[N1-1] - b[N1-1];
        a[N1]   = a[N1]   - b[N1-1];
        form_Q(N1,a,b,d,Q1);
        form_Q(N2,&a[N1],&b[N1],&d[N1],Q2);

        
        cnt = 0;
        for(i=0;i<N1;i++)
          xi[cnt++] = Q1[N1-1][i];
        for(i=0;i<N2;i++)
          xi[cnt++] = Q2[0][i];

		//printf("About to solve secular equation\n");
        bisection(b[N1-1],N, d, xi, lambda);

		printf("Forming Q\n");
        for(i=0;i<N1;i++){
            for(j=0;j<N;j++){
                Q[i][j] = 0.0;
                for(k=0;k<N1;k++)
                    Q[i][j] += Q1[i][k]*xi[k]/(d[k]-lambda[j]);
            }
        }
        for(i=0;i<N2;i++){
            for(j=0;j<N;j++){
                Q[N1+i][j] = 0.0;
                for(k=0;k<N2;k++)
                    Q[i+N1][j] += Q2[i][k]*xi[N1+k]/(d[N1+k]-lambda[j]);
            }
        }

        double sum;
        for(i=0;i<N;i++){
            sum = 0.0;
            for(j=0;j<N;j++)
                sum+= Q[j][i]*Q[j][i];
            sum = sqrt(sum);
            for(j=0;j<N;j++)
                Q[j][i] = Q[j][i]/sum;
        }

		// if(iam==0) {
		// 	for(i=0; i<N; i++) {
		// 		for(j=0;j<N;j++) {
		// 			printf("%f \n",Q[i][j]);
		// 		}
		// 	}
		// }		

        sdestroy_1d(d);
        sdestroy_1d(xi);
        sdestroy_2d(Q1, N1);
        sdestroy_2d(Q2, N2);
    }
    return;
}

/*------------------------------------------------------------------------
    Solve the secular equation
--------------------------------------------------------------------------*/
void bisection(double bm, int n, double *d, double *xi, double *lambda)
{
    double xl,xr,xm;
    double yl,yr,ym;
    const double offset = 1.0e-5;
    const double tol = 1.0e-6;

    int i;
    for(i=0; i<(n-1); i++) {
        xl = d[i] + offset;
        yl = secular (bm, n, d, xi, xl);
        xr = d[i+1] - offset;
        yr = secular (bm, n, d, xi, xr);
        xm = (xl + xr)/2.;
        ym = secular (bm, n, d, xi, xm);

        if(yl * yr > 0) {
            lambda[i] = xl;
            continue;
        }

        while (fabs(ym) > tol) {
            if(yl * ym < 0) xr = xm;
            else xl = xm;
            xm = (xl + xr)/2.;
            ym = secular (bm, n, d, xi, xm);
        }
        lambda[i] = xm;
    }

    xl = d[n-1] + offset;
    yl = secular (bm, n, d, xi, xl);
    xr = 2 * d[n-1];
    yr = secular (bm, n, d, xi, xr);
    xm = (xl + xr)/2.;
    ym = secular (bm, n, d, xi, xm);

    if(yl * yr > 0) {
        lambda[n-1] = xl;
    } else {
        while (fabs(ym) > tol) {
            if(yl * ym < 0) xr = xm;
            else xl = xm;
            xm = (xl + xr)/2.;
            ym = secular (bm, n, d, xi, xm);
        }
        lambda[n-1] = xm;
    }
}


/*------------------------------------------------------------------------
    Functional form of the secular equation
--------------------------------------------------------------------------*/
double secular(double bm, int n, double *d, double *xi, double x) 
{
    double sum = 0.;
    for(int i=0; i<n; i++)
        sum += xi[i]*xi[i]/(d[i]-x);
    sum = bm * sum + 1.e0;
    return sum;
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int nlocal, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j) {
        /* Compute C(i,j) */
        double cij = C[i+j*lda];
        for (int k = 0; k < K; ++k)
            cij += A[i+k*lda] * B[k+j*lda];
        C[i+j*lda] = cij;
    }
}

/*------------------------------------------------------------------------
    Blocked transpose matrix multiplication
--------------------------------------------------------------------------*/
int matrixMultiply(double *a, double *b, double *c, int n, int n_local) 
{
    int i, j, k;
    double sum;
    for (i=0; i<n_local; i++) {
         for (j=0; j<n; j++) {
             sum = 0.;
             for (k=0; k<n; k++) {
                 sum += a[i*n + k] * b[k*n + j];
             }
             c[i*n + j] = sum;
         }
    }
    return 0;

    /* For each block-row of A */
    //for (int i = 0; i < n_local; i += BLOCK_SIZE)
        /* For each block-column of B */
     //   for (int j = 0; j < n; j += BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
     //       for (int k = 0; k < n; k += BLOCK_SIZE) {
                 //Correct block dimensions if block "goes off edge of" the matrix 
     //           int M = min (BLOCK_SIZE, n_local-i);
     //           int N = min (BLOCK_SIZE, n-j);
     //           int K = min (BLOCK_SIZE, n-k);

                /* Perform individual block dgemm */
    //            do_block(n, n, M, N, K, a + i + k*n, b + k + j*n, c + i + j*n);
    //        }
    //return 0;
}

/*------------------------------------------------------------------------
    Create real, symmetric 1D array. Diagonally dominant
    Column major ordering
--------------------------------------------------------------------------*/
void sym_matrix(int Adim, double *Asym)
{
    srand(time(NULL));
    double aij, aii;
    for(int ii=0; ii<Adim; ii++) {
        for(int jj=0; jj<Adim; jj++) {
            aij = (rand()/((double) RAND_MAX))/10;
			if(ii==jj) {
				Asym[ii+jj*Adim] = ii + jj + 2.;
			} else {
            	Asym[ii+jj*Adim] = aij;
            	Asym[jj+ii*Adim] = aij;
			}
        }
    }
}

/*------------------------------------------------------------------------
    Allocate memory for a 1D doubple precision array, initialized to 0.
--------------------------------------------------------------------------*/
double *allocate_1d(int rows, int cols)
{
    double *arr = (double *)calloc(rows*cols,sizeof(double)) ;
    return arr;
}

/*------------------------------------------------------------------------
    Allocate memory for a 1D doubple precision array, initialized to 0.
--------------------------------------------------------------------------*/
double *allocate_vector(int length)
{
    double *vec = (double *)calloc(length,sizeof(double)) ;
    return vec;
}

/*------------------------------------------------------------------------
    Allocate memory for a 1D doubple precision array, initialized to 0.
--------------------------------------------------------------------------*/
double **allocate_2d(int rows, int cols)
{
    double **arr = new double*[rows];

    for(int ii=0;ii<rows;++ii) arr[ii] = new double[cols];

    for(int jj=0;jj<rows;jj++) {
        for(int kk=0;kk<cols;kk++) {
            arr[jj][kk] = 0.;
        }
    }
    return arr;
}

/*------------------------------------------------------------------------
    Allocate memory for a 1D doubple precision array, initialized to 0.
--------------------------------------------------------------------------*/
int *allocate_int_1d(int rows, int cols)
{
    int *arr = (int *)calloc(rows*cols,sizeof(int)) ;
    return arr;
}

/*------------------------------------------------------------------------
    Allocate memory for a 1D doubple precision array, initialized to 0.
--------------------------------------------------------------------------*/
int **allocate_int_2d(int rows, int cols)
{
    int **arr = new int*[rows];

    for(int ii=0;ii<rows;++ii) arr[ii] = new int[cols];

    for(int jj=0;jj<rows;jj++) {
        for(int kk=0;kk<cols;kk++) {
            arr[jj][kk] = 0.;
        }
    }
    return arr;
}

/*------------------------------------------------------------------------
    Construct 1D identity matrix, double precision
    column-major
--------------------------------------------------------------------------*/
double *identity_1d(int rows, int cols)
{
    double *arr = (double *)calloc(rows*cols,sizeof(double)) ;

    for(int jj=0;jj<rows;jj++) {
        for(int kk=0;kk<cols;kk++) {
            arr[jj + rows * kk] = jj==kk ? 1 : 0;
        }
    }
    return arr;
}

/*------------------------------------------------------------------------
    Safe method of freeing memory
--------------------------------------------------------------------------*/
void sdestroy_1d(double *matrix)
{
    if(matrix == NULL) return;
    free(matrix);
}

/*------------------------------------------------------------------------
    Safe method of freeing 2D memory
--------------------------------------------------------------------------*/
void sdestroy_2d(double **matrix, int width)
{
    if(matrix == NULL) return;

    for(int ii=0; ii<width; ii++) {
        delete [] matrix[ii];
    }
    delete [] matrix;
}

/*------------------------------------------------------------------------
    Kill program, prints a message along with line number and file 
    where error occurs.
--------------------------------------------------------------------------*/
void kill(const char *file, int line, const char *message)
{
    if(message!=NULL)
    fprintf(stderr,"  %s >> (%s:%d)\n",message,file,line);
    exit (EXIT_FAILURE);
}


/*------------------------------------------------------------------------
	Funtion to map cpu id
	'dest' OPTION IN MPI_Send MUST BE AN INTEGER BETWEEN 0 AND 'n_proc - 1' 
--------------------------------------------------------------------------*/
int map_processors(int i, int N, int n_procs) 
{
  n_procs = n_procs - 1;
  int r = (int) ceil( (double) N / (double) n_procs);
  int processor = i / r;
 
  return processor + 1;
}

