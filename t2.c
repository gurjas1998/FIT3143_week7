#include <stdio.h>
#include "mpi.h"
/* This example handles a 12 x 12 mesh, on 4 processors only. */
#define maxn 12
int main()     
    {
        int rank, value, size, errcnt, toterr, i, j;
        int up_nbr, down_nbr;
        MPI_Status status;
        double x[12][12];
        double xlocal[(12/4)+2][12];
        MPI_Init(NULL, NULL);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        
        if (size != 4) MPI_Abort( MPI_COMM_WORLD, 1 );
        
        /* xlocal[][0] is lower ghostpoints, xlocal[][maxn+2] is upper */
        /* Fill the data as specified */
        /*For rows 1 to 3, fill in all the elements with the rank of the process*/
        for (i=1; i<=maxn/size; i++) for (j=0; j<maxn; j++) xlocal[i][j] = rank;
        
        /* Fill in the ghostpoints (ie, rows 0 and last) with -1. This might be unnecessary since 
        everything gets replaced later BUT it's relevant for process 0's row 0 and process 3's row 4 */
        for (j=0; j<maxn; j++) {
            xlocal[0][j] = - 1;
            xlocal[maxn/size+1][j] = - 1;
        }
        
        /* Setting up_nbr and down_nbr to be the processes of higher and lower rank
        that we will be communicating with */
        /* Note that we use MPI_PROC_NULL to remove the if statements that would be needed without MPI_PROC_NULL
        This is necessary because rank 0 and 3 should NOT be sending/receiving to/from each other*/
        up_nbr = rank + 1;
        if (up_nbr >= size) up_nbr = MPI_PROC_NULL;
        down_nbr = rank - 1;
        if (down_nbr < 0) down_nbr = MPI_PROC_NULL;
        
        /* Starting the Sendrecvs here */
        /* Send up and receive from below (shift up) */
        /* Note the use of xlocal[i] for &xlocal[i][0] */
        MPI_Sendrecv( xlocal[maxn/size], maxn, MPI_DOUBLE, up_nbr, 0, xlocal[0], maxn, MPI_DOUBLE, down_nbr, 0, MPI_COMM_WORLD, &status );
        /* Send down and receive from above (shift down) */
        MPI_Sendrecv( xlocal[1], maxn, MPI_DOUBLE, down_nbr, 1, xlocal[maxn/size+1], maxn, MPI_DOUBLE, up_nbr, 1, MPI_COMM_WORLD, &status );
        
        /* Check that we have the correct results */
        errcnt = 0;
        /* If anything from rows 1 to 3 do not match the processor's rank, it's an error */
        for (i=1; i<=maxn/size; i++) for (j=0; j<maxn; j++) if (xlocal[i][j] != rank) errcnt++;
        
        /* Check that row 0 values should be equivalent to rank - 1 (ie. sent from
        the processor with lower rank), and row 3 values equiv to rank + 1.
        We also ignore row 3 ghostpoints for process rank 3 because it'll be -1 */
        for (j=0; j<maxn; j++) {
            if (xlocal[0][j] != rank - 1) errcnt++;
            if (rank < size - 1 && xlocal[maxn/size+1][j] != rank + 1) errcnt++;
        }
        
        /* Sums up the errcnt in each process and stores in toterr */
        MPI_Reduce( &errcnt, &toterr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
        
        if (rank == 0) {
            if (toterr) printf( "! found %d errors\n", toterr );
        else printf( "No errors\n" );
        }
        MPI_Finalize();
        return 0;
        }
