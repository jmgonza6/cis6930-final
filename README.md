# TRQR (Tri-Diagonal Reduction + QR factorization) V. 0.6,  4.23.2015
___________________________________________________________________________________

TDRQR:
------
	'TDRQR' is a routine for computing the eigenvalues of a symmmetric matrix.
The routine is based on the idea that diagonalizing a matrix is O(n) if the 
matrix is in tri-diagonal form.  Therefore, the input matrix is first reduced
to tri-diagonal form via successive Householder transformations. The resulting 
matrix is stored in 1d vectors since only the diagonal elements survive.  This 
matrix is sent amongst the cpus, and diagonalized lcoally.  The main MPI
collectives used in this algorithm are simple send and recv, with a few bcasts.

___________________________________________________________________________________

Usage:
------
'TDRQR' is run very simply, with mpirun -n N ./tdrqr -n 500

A few options are available:

-n 		Dimension of matrix

-d 		Debug flag, print all MPI Collectives as well as the input and intermediate
		matrices/vectors

-verify	Runs a test case, 4x4 matrix with known solution

NOTE: The number of processors needs to be a power of 2.  
	  Will be fixed in future releases.


___________________________________________________________________________________

Author:
-------
Comments and bugs are appreciated!

    /*
    Joseph M. Gonzalez 
    Materials Simulation Laboratory
    Department of Physics, University of South Florida, Tampa, FL 33620
    jmgonza6@mail.usf.edu
    */

___________________________________________________________________________________

Installation:
-------------
Edit the supplied Makefile according to your MKL/BLAS libraries.

If you do not have access to BLAS, then a custom vectorized matrix multiplication
subroutine, written in ANSI C, can be used.  Just replace -DBLAS with -DJMG in the 
Makefile.

___________________________________________________________________________________

Copyright:
----------
Copyright (C) 2015 Joseph M. Gonzalez

'TDRQR' is software which is distributed under the GNU General Public License.

'TDRQR' is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

___________________________________________________________________________________

Contents:
---------
    README.md   --> this file
    Makefile	--> Build rules
    LICENSE     --> GNU General public license
    utils.h     --> Header file with all fucntion prototypes and MACROS
    utils.cpp   --> Implementation of auxillary functions used by main.cpp
    main.cpp    --> Main driver, handles MPI send/recv. No comments
    			    Calls subroutines from utils.cpp/h
    main-comments.cpp self explanitory
