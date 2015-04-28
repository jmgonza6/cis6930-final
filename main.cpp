/*************************************************************************************
* FILE: main.cpp
* DESCRIPTION:  
*   'TRDQR' is a routine for computing the eigenvalues of a symmmetric matrix.
*    The routine is based on the idea that diagonalizing a matrix is O(n) if the 
*    matrix is in tri-diagonal form.  Therefore, the input matrix is first reduced
*    to tri-diagonal form via successive Householder transformations. The resulting 
*    matrix is stored in 1d vectors since only the diagonal elements survive.  This 
*    matrix is sent amongst the cpus, and diagonalized lcoally.  The main MPI
*    collectives used in this algorithm are simple send and recv, with a few bcasts.
*
* AUTHOR: Joseph Gonzalez, 
*         PhD Student, Materials Simulation Laboratory
*         Department of Physics, University of South Florida, Tampa, FL 33620
*         web: msl.cas.usf.edu
*         Ph: 813-380-7137 
* LAST REVISED: 04/24/2015
*************************************************************************************/

#include "utils.h"

int main(int argc, char **argv) 
{
    MPI::Init(argc,argv);
    int iam = MPI::COMM_WORLD.Get_rank( );
    int world = MPI::COMM_WORLD.Get_size( );
    MPI::Status status;
    int sent_msg, msg_size;

    bool debug = false, verify = false;
    char message[128];
    int bytes_allocated = 0;

    int n = 0, n_ubound, n_local, n_sq;

    int i, j, ii, jj, kk, ll, m, k;

    int isum,offset, n1, n2, p2, count;
    double bn, *tmpd, **tmpdd;

    double dgemm_time,send_time,recv_time, Q_time;
    double start_dgemm, end_dgemm =0,
            start_send, end_send =0,
            start_recv, end_recv =0,
            start_run, end_run =0,
            start_Q, end_Q,
            runtime;

    
    if (iam==MASTER) {
        for( int ii = 1; ii < argc; ii++ ) {
            if( strcmp( argv[ii], "-n" ) == 0 ) {
                n = atoi(argv[ii+1]);
            } else if (strcmp( argv[ii], "-d" ) == 0 ) {
                debug = true;
            } else if( strcmp( argv[ii], "-verify" ) == 0 ) {
                verify = true;
                debug = true;
                n = 4;
            }
        }
    }

    MPI::COMM_WORLD.Bcast(&n, 1, MPI::INT, MASTER);
    if (!n) {
        sprintf(message,"No dimension specified!!\n");
        kill(FLERR,message);
        MPI::Finalize();
    }

    n_local = getRowCount(n, iam, world);

    n_ubound = n * n_local;
    n_sq     = n * n;

    if (iam==MASTER) {
        printf("%s has started with %d tasks.\n",SOFTN,world);
        printf("Initializing arrays...\n");
    }

    double *A = allocate_1d(n,n);
    bytes_allocated += n * n * sizeof(double);
    
    if (iam==MASTER) {
        if(verify) {
            A[0 + 0*n] = 4;
            A[0 + 1*n] = 2;
            A[0 + 2*n] = 2;
            A[0 + 3*n] = 1;

            A[1 + 0*n] = 2;
            A[1 + 1*n] = -3;
            A[1 + 2*n] = 1;
            A[1 + 3*n] = 1;

            A[2 + 0*n] = 2;
            A[2 + 1*n] = 1;
            A[2 + 2*n] = 3;
            A[2 + 3*n] = 1;

            A[3 + 0*n] = 1;
            A[3 + 1*n] = 1;
            A[3 + 2*n] = 1;
            A[3 + 3*n] = 2;
        } else {
            sym_matrix(n,A);
        }
        bytes_allocated += n * n * sizeof(double);
    }

    int iter;
    int ops = n-2;
    
    double *tmpA = A;

    #ifdef __LINUX__
    char hostname[1024];
    gethostname(hostname, 1024);
    #endif
    time_t now;
    time(&now);
    char fname[128];
    sprintf(fname,"%s_%d_%i.log",SOFTN,world,n);
    FILE *logfile = fopen(fname,"w"); 
    if(iam==MASTER) {
        int wun = 1;
        fprintf(logfile, "------------------------------------------------------------\n");
        #ifdef __LINUX__
        gethostname(hostname, 1024);
        fprintf(logfile, " executed on host: %s\n", hostname);
        #endif
        fprintf(logfile, " Date: %s\n", ctime(&now));
        fprintf(logfile, " %s %s\n", SOFTN, SOFTV);
        fprintf(logfile, " Build: %s, %s\n",CVERSION,DATE);
        fprintf(logfile, "        %s\n",BUILD);
        fprintf(logfile, "------------------------------------------------------------\n");
        fprintf(logfile, " MPI Tasks:                                  %3i\n",world);
        fprintf(logfile, " Matrix dimensions:                         %4i x %4i\n",n,n);
    }

    start_run = MPI::Wtime();
    for(iter = 0; iter < ops; iter++) {
        double *H = allocate_1d(n, n);
        double *HA = allocate_1d(n, n);

        if (iam==MASTER) {
            house(H, tmpA, n, iter);
            sent_msg = n_ubound;
            start_send = MPI::Wtime();
            for (i=1; i<world; i++) {
                msg_size = n * getRowCount(n, i, world);
                if(debug) printf("Sending a message of %d ints from A to task %d\n",msg_size,i);
                MPI::COMM_WORLD.Send(H + sent_msg,msg_size,MPI::DOUBLE,i,TAG_INIT);
                
                sent_msg += msg_size;
            }
            end_send = end_send + MPI::Wtime() - start_send;
        } else { 
            if(debug) printf("Receiving a message of %d ints from cpu %d\n",msg_size,MASTER);
            start_recv = MPI::Wtime();
            MPI::COMM_WORLD.Recv(H, n_ubound,MPI::DOUBLE, MASTER,TAG_INIT, status); 
            end_recv = end_recv + MPI::Wtime() - start_recv;
        }

        if(debug) printf("Broadcasting tmpA all procs\n");
        MPI::COMM_WORLD.Bcast(tmpA, n*n, MPI::DOUBLE, MASTER);

        if(debug) printf("Computing dgemm\n");

        start_dgemm = MPI::Wtime();
        matrixMultiply(H, tmpA, HA, n, n_local);
        end_dgemm = end_dgemm + MPI::Wtime() - start_dgemm;


        if (iam==MASTER) {
            sent_msg = n_ubound;
            start_recv = MPI::Wtime();
            for (i=1; i<world; i++) {
                msg_size = n * getRowCount(n, i, world);
                if(debug) printf("Receiving a message of %d ints from cpu %d\n",msg_size,i);
                MPI::COMM_WORLD.Recv(HA + sent_msg, msg_size,  MPI::DOUBLE, i,TAG_RESULT,status);
                
                sent_msg += msg_size;
            }
            end_recv = end_recv + MPI::Wtime() - start_recv;
        } else { 
            start_send = MPI::Wtime();
            MPI::COMM_WORLD.Send(HA, n_ubound,MPI::DOUBLE, MASTER, TAG_RESULT);
            end_send = end_send + MPI::Wtime() - start_send;
        }

        if (iam==MASTER) {
            sent_msg = n_ubound;
            start_send = MPI::Wtime();
            for (i=1; i<world; i++) {
                msg_size = n * getRowCount(n, i, world);
                if(debug) printf("Sending a message of %d ints from A to task %d\n",msg_size,i);
                MPI::COMM_WORLD.Send(HA + sent_msg,msg_size, MPI::DOUBLE,i,TAG_INIT);
                
                sent_msg += msg_size;
            }
            end_send = end_send + MPI::Wtime() - start_send;
        } else { 
            if(debug) printf("Sending a message of %d ints to cpu %d\n",msg_size,i);
            start_recv = MPI::Wtime();
            MPI::COMM_WORLD.Recv(HA,n_ubound, MPI::DOUBLE, MASTER,TAG_INIT,status);
            end_recv = end_recv + MPI::Wtime() - start_recv;
        }

        if(debug) printf("Broadcasting H all procs\n");
        MPI::COMM_WORLD.Bcast(H, n*n, MPI::DOUBLE, MASTER);

        if(debug) printf("Computing dgemm\n");
        start_dgemm = MPI::Wtime();
        matrixMultiply(HA, H, tmpA, n, n_local);
        end_dgemm = end_dgemm + MPI::Wtime() - start_dgemm;

        if (iam==MASTER) {
            sent_msg = n_ubound;
            start_recv = MPI::Wtime();
            for (i=1; i<world; i++) {
                msg_size = n * getRowCount(n, i, world);
                if(debug) printf("Receiving a message of %d ints from cpu %d\n",msg_size,i);
                MPI::COMM_WORLD.Recv(tmpA + sent_msg,msg_size,MPI::DOUBLE,i,TAG_RESULT,status);
                sent_msg += msg_size;
            }
            end_recv = end_recv + MPI::Wtime() - start_recv;
        } else { 
            start_send = MPI::Wtime();
            MPI::COMM_WORLD.Send(tmpA,n_ubound,MPI::DOUBLE,MASTER,TAG_RESULT);
            end_send = end_send + MPI::Wtime() - start_send;
        }
        sdestroy_1d(H);
        sdestroy_1d(HA);
    }

    if(iam==MASTER) {
        send_time = end_send;
        recv_time = end_recv;
        dgemm_time = end_dgemm;
        fprintf(logfile, "------------------------------------------------------------\n");
        fprintf(logfile, " Householder:\n");
        fprintf(logfile, "   CPU Time spent sending matrices (sec):       %8.6f\n",send_time);
        fprintf(logfile, "   CPU Time spent in dgemm (sec):               %8.6f\n",dgemm_time);
        fprintf(logfile, "   CPU Time spent receiving matrices (sec):     %8.6f\n",recv_time);
    }

    double *a = allocate_vector(n);
    double *b = allocate_vector(n);
    bytes_allocated += 2 * n * sizeof(double);
    
    double **Q =  allocate_2d(n,n); 
    double **Q1 = allocate_2d(n,n);
    double **Q2 = allocate_2d(n,n);

    double *xi = allocate_vector(n);
    double *d  = allocate_vector(n);
    double *lambda = allocate_vector(n);
    bytes_allocated += 3 * n * sizeof(double);
    bytes_allocated += 3 * n * n * sizeof(double);


    int **cpu_map = allocate_int_2d(world,world);
    double **adjust = allocate_2d(world,world);
    bytes_allocated += world * world * sizeof(double);
    bytes_allocated += world * world * sizeof(int);
    
    for(i=0;i<n;i++){
        a[i] = i +1.5;
        b[i] = 0.3;
    }

    // for(ii=0;ii<n;ii++) {
    //     for(kk=0;kk<n;kk++) {
    //         if(ii==kk) {
    //             a[ii] = tmpA[ii + n * kk];
    //             b[ii] = tmpA[(ii + n * kk)+1];
    //         }
    //     }
    // }

    if(debug) printf("Broadcasting a all procs\n");
    MPI::COMM_WORLD.Bcast(a, n, MPI::DOUBLE, MASTER);
    if(debug) printf("Broadcasting b all procs\n");
    MPI::COMM_WORLD.Bcast(b, n, MPI::DOUBLE, MASTER);

    int lg2 = log2(world);
    cpu_map[0][0] = n; 
    for(i=0;i<lg2;i++){
        isum = 0;
        p2 = pow(2.0,i);
        for(j=0;j<p2;j++){
            cpu_map[i+1][2*j] = cpu_map[i][j]/2;
            cpu_map[i+1][2*j+1] = cpu_map[i][j] - cpu_map[i+1][2*j];
            isum += cpu_map[i+1][2*j];
            adjust[i][j] = b[isum-1];
            a[isum-1] = a[isum-1] - b[isum-1];
            a[isum]   = a[isum]   - b[isum-1];
            isum += cpu_map[i+1][2*j+1];
        }
    }

    offset = lg2;
    isum = 0;
    for(k=0;k<iam;k++)
        isum += cpu_map[offset][k];

    form_Q(cpu_map[offset][iam],&a[isum],&b[isum],d,Q1);

    for(i=0;i<lg2;i++){
        isum = 0; count = 0;
        p2 = pow(2.0,i);
        for(j=0;j<world;j+=p2){
            if(iam == j){
                if(isum%2==0){
                    start_recv = MPI::Wtime();
                    MPI::COMM_WORLD.Recv(d+cpu_map[offset][isum],cpu_map[offset][isum+1],MPI::DOUBLE,j+p2,TAG_RESULT,status);
                    for(k=0;k<cpu_map[offset][isum+1];k++) {
                        MPI::COMM_WORLD.Recv(Q2[k],cpu_map[offset][isum+1], MPI::DOUBLE,j+p2, TAG_RESULT, status);
                    }
                    end_recv = end_recv + MPI::Wtime() - start_recv;
                    n1 = cpu_map[offset][isum];
                    n2 = cpu_map[offset][isum+1];
                    bn = adjust[offset-1][count++];
                } else{
                    start_send = MPI::Wtime();
                    MPI::COMM_WORLD.Send(d, cpu_map[offset][isum], MPI::DOUBLE,j-p2,TAG_RESULT);
                    for(k=0;k<cpu_map[offset][isum];k++) {
                        MPI::COMM_WORLD.Send(Q1[k],cpu_map[offset][isum],MPI::DOUBLE,j-p2,TAG_RESULT);               
                    }
                    end_send = end_send + MPI::Wtime() - start_send;
                }
            }
            isum++;
        }

        p2 = pow(2.0,i+1);
        for(j=0;j<world;j+=p2){
            if(iam == j){
                count = 0;
                for(k=0;k<n1;k++)
                    xi[count++] = Q1[n1-1][k];
                for(k=0;k<n2;k++)
                    xi[count++] = Q2[0][k];

                bisection (bn,n1+n2,d,xi,lambda);

                for(k=0;k<n1;k++){
                    for(ll=0;ll<n1+n2;ll++){
                        Q[k][ll] = 0.0;
                        for(m=0;m<n1;m++)
                            Q[k][ll] += Q1[k][m]*xi[m]/(d[m]-lambda[ll]);
                        }
                }
                for(k=0;k<n2;k++){
                    for(ll=0;ll<n1+n2;ll++){
                        Q[n1+k][ll] = 0.0;
                    for(m=0;m<n2;m++)
                        Q[k+n1][ll] += Q2[k][m]*xi[n1+m]/(d[n1+m]-lambda[ll]);
                    }
                }

                double sum;
                for(k=0;k<n1+n2;k++){
                    sum = 0.0;
                    for(ll=0;ll<n1+n2;ll++)
                        sum+= Q[ll][k]*Q[ll][k];
                    sum = sqrt(sum);
                    for(ll=0;ll<n1+n2;ll++)
                        Q[ll][k] = Q[ll][k]/sum;
                }

                tmpd = d;
                d = lambda;
                lambda = tmpd;

                tmpdd = Q1;
                Q1 = Q;
                Q = tmpdd;
            }          
        }
        offset = offset - 1;
    } 

    if(iam==MASTER) {
            send_time = end_send;
            recv_time = end_recv;
            Q_time = end_Q;
            fprintf(logfile, "------------------------------------------------------------\n");
            fprintf(logfile, " Forming Q, Secular:\n");
            fprintf(logfile, "   CPU Time spent sending matrices (sec):       %8.6f\n",send_time);
            fprintf(logfile, "   CPU Time spent on diagonalization (sec):     %8.6f\n",Q_time);
            fprintf(logfile, "   CPU Time spent receiving matrices (sec):     %8.6f\n",recv_time);
    }

   if(iam==MASTER) {
       printf("The eigenvalues are: \n");
       for(k=0;k<n;k++)
           printf("%f\n",d[k]);

   }

    end_run = MPI::Wtime() - start_run;
    double compute_time = dgemm_time;
    double system_time = end_run - compute_time ;

    int mem_bytes =0;
    MPI::COMM_WORLD.Reduce(&bytes_allocated, &mem_bytes,1, MPI::INT, MPI::SUM, MASTER);            
                          
    double total_mem = (double)(mem_bytes*(1.e-6));

    if(iam==MASTER) {
        fprintf(logfile, "------------------------------------------------------------\n");
        fprintf(logfile, " Run-time accounting:\n");
        fprintf(logfile, "   Total memory used by root process (MB):      %8.6f\n",(double)(bytes_allocated*(1.e-6)));
        fprintf(logfile, "   Total memory used (MB):                      %8.6f\n\n",total_mem);
        fprintf(logfile, "   Computation time (sec):                      %8.6f\n",compute_time);
        fprintf(logfile, "   System time (sec):                           %8.6f\n",system_time);
        fprintf(logfile, "                                               ----------\n");
        fprintf(logfile, "   Total CPU time (sec):                        %8.6f\n",end_run);
        fprintf(logfile, "------------------------------------------------------------\n");
    }
    fclose(logfile);

    if (iam==MASTER&&debug) {
        for (i=0; i<n; i++) {
            for(j=0;j<n;j++) {
                printf("%f ",A[i*n + j]);
            }
            printf("\n");
        }

        printf("\n");
        for (i=0; i<n; i++) {
            for(j=0;j<n;j++) {
                printf("%f ",tmpA[i*n + j]);
            }
            printf("\n");
        }
    }
    printf("\n");


    MPI::COMM_WORLD.Barrier();

    sdestroy_1d(A);

    sdestroy_1d(lambda);
    sdestroy_1d(xi);
    sdestroy_1d(d);
    sdestroy_1d(a);
    sdestroy_1d(b);

    sdestroy_2d(Q,n);
    sdestroy_2d(Q1,n);
    sdestroy_2d(Q2,n);
 
    MPI::Finalize();
    return 0;
}


