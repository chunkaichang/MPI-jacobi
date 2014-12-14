#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

//Reference implementation of Jacobi for mp2 (sequential)
//Constants are being used instead of arguments
#define BC_HOT  1.0
#define BC_COLD 0.0
#define INITIAL_GRID 0.5
#define TOL 1.0e-8
#define ARGS 5

struct timeval tv;
double get_clock() {
   struct timeval tv; int ok;
   ok = gettimeofday(&tv, (void *) 0);
   if (ok<0) { printf("gettimeofday error");  }
   return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


double **create_matrix(int n) {
	int i;
	double **a;

	a = (double**) malloc(sizeof(double*)*n);
	for (i=0;i<n;i++) {
		a[i] = (double*) malloc(sizeof(double)*n);
	}

	return a;
}

void init_matrix(double **a, int n) {

	int i, j;
	
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++)
			a[i][j] = INITIAL_GRID;
	}
}

void swap_matrix(double ***a, double ***b) {

	double **temp;

	temp = *a;
	*a = *b;
	*b = temp;	
}

void print_grid(double **a, int nstart, int nend) {

	int i, j;

	for(i=nstart; i<nend; i++) {
		for(j=nstart; j<nend; j++) {
			printf("%6.4lf ",a[i][j]);
		}
		printf("\n");
	}
}

void free_matrix(double **a, int n) {
	int i;
	for (i=0;i<n;i++) {
		free(a[i]);
	}
	free(a);
}

int main(int argc, char* argv[]) {
	int i,j,n,r,c,iteration,bclength,max_iterations;
	double **a, **b, maxdiff;
	double tstart, tend, ttotal;

	if (argc != ARGS) {
		fprintf(stderr,"Wrong # of arguments.\nUsage: %s N I R C\n",
					argv[0]);
		return -1;
	}
	
	
	n = atoi(argv[1]);
	max_iterations = atoi(argv[2]);
	r = atoi(argv[3]);
	c = atoi(argv[4]);

	
	//add 2 to each dimension to use sentinal values
	a = create_matrix(n+2);
	b = create_matrix(n+2);

	init_matrix(a,n+2);

	bclength = (n+2)/2;
	
	//Initialize the hot boundary
	for(i=0;i<bclength;i++) {
		a[i][0] = BC_HOT;
	}

	// Initialize the cold boundary
	for(j=n+2-bclength;j<n+2;j++) {
		a[n+1][j] = BC_COLD;
	}

	// Copy a to b
	for(i=0; i<n+2; i++) {
		for(j=0; j<n+2; j++) {
			b[i][j] = a[i][j];
		}
	}

	// Output initial grid
	//printf("Initial grid:\n");
	//print_grid(a,1,n+1);
	
	// MPI Initialization
	int rank,size,stripSize,offset,ierr;
	MPI_Request send_request,send_request_2,recv_request,recv_request_2;
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	stripSize = n / size;
	
	// Main simulation routine
	iteration=0;
	maxdiff=1.0;
	if(rank == 0){
	printf("Running simulation with tolerance=%lf and max iterations=%d\n",
		TOL, max_iterations);
		
	tstart = get_clock();
	}

	while(maxdiff > TOL && iteration<max_iterations) {

		// Initialize boundary values
		// Top
		if(rank == 0){
		    for(j=0; j<n+2; j++)
			a[0][j] = a[1][j];
		}

		// Bottom
		if(rank == size-1){
		    for(j=0; j<n+2-bclength; j++)
			a[n+1][j] = a[n][j];
		}

		// Left
		for(i=bclength; i<n+2; i++)
			a[i][0] = a[i][1];
		// Right
		for(i=0; i<n+2; i++)
			a[i][n+1] = a[i][n];

		// Compute new grid values
		maxdiff = 0.0;

		double local_maxdiff = 0.0;

		if(rank == 0){
		    // Send last row to rank 1
		    MPI_Isend(a[stripSize], n+2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &send_request);
		    
		    // Calculate rest parts
		    for(i=1; i < stripSize; i++){

		        for(j=1; j < n+1; j++){
			    b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
				
			    if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
				local_maxdiff = fabs(b[i][j]-a[i][j]);
			}

		    }
	 	    // Recieve a neighbor row from rank 1
		    MPI_Irecv(a[stripSize+1], n+2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &recv_request);
		    
		    MPI_Wait(&recv_request,&status);
		    // Calculate the last row
		    i = stripSize;

		    for(j=1; j < n+1; j++){
			b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
			   
			if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			    local_maxdiff = fabs(b[i][j]-a[i][j]);
		    }
		    
		    MPI_Wait(&send_request,&status);

		}
		else if(rank == (size - 1) ){
		    // Send first row to rank size-2
		    MPI_Isend(a[rank * stripSize + 1], n+2, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD, &send_request);
		    
		    // Calculate rest parts
		    for(i= n - stripSize + 2 ; i < n+1; i++){

		        for(j=1; j < n+1; j++){
			    b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
			
			    if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			        local_maxdiff = fabs(b[i][j]-a[i][j]);
			}    
		    }	    
	 	    
		    // Recieve a neighbor row from rank size-2
		    MPI_Irecv(a[n - stripSize], n+2, MPI_DOUBLE, size - 2, 0, MPI_COMM_WORLD, &recv_request);
		    
		    MPI_Wait(&recv_request,&status);
		    // Calculate the first row
		    i = n - stripSize + 1;

		    for(j=1; j < n+1; j++){
			b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
		    
			if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			    local_maxdiff = fabs(b[i][j]-a[i][j]);
		    }
		    MPI_Wait(&send_request,&status);
		}
		else{
		    // Send first row to rank - 1
		    MPI_Isend(a[rank * stripSize + 1], n+2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,&send_request);

		    // Send last row to rank + 1
		    MPI_Isend(a[(rank+1) * stripSize], n+2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,&send_request_2);

		    // Calculate rest parts
		    for(i= rank * stripSize + 2; i < (rank + 1)*stripSize -1; i++){

		        for(j=1; j < n+1; j++){
			    b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
			
			    if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			        local_maxdiff = fabs(b[i][j]-a[i][j]);
			}    
		    }	    

		    // Recieve a neighbor row from rank - 1
		    MPI_Irecv(a[rank * stripSize], n+2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request);

		    // Recieve a neighbor row from rank + 1
		    MPI_Irecv(a[(rank+1) * stripSize + 1], n+2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request_2);
		    
		    // Wait until recv
		    MPI_Wait(&recv_request,&status);
		    
		    // Calculate first row
		    i = rank * stripSize + 1; 

		    for(j=1; j < n+1; j++){
			b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
		        
			if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			        local_maxdiff = fabs(b[i][j]-a[i][j]);
		    }
		    
		    // Wait until recv
		    MPI_Wait(&recv_request_2,&status);
		    
		    // Calculate the last row
		    i = (rank+1)*stripSize;

		    for(j=1; j < n+1; j++){
			b[i][j] = 0.2*(a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1]);
			    
			if(fabs(b[i][j]-a[i][j]) > local_maxdiff)
			    local_maxdiff = fabs(b[i][j]-a[i][j]);
		    }	    
		    MPI_Wait(&send_request,&status);
		    MPI_Wait(&send_request_2,&status);
		}

		// Reduce all local_maxdiff to the global maxdiff
		MPI_Allreduce(&local_maxdiff, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
		// Copy b to a
		swap_matrix(&a,&b);	

		iteration++;

		// Barrier all ranks
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(rank == 0)
	{
	tend = get_clock();
	ttotal = tend-tstart;
	}
	//Rank 0 recieve data from others
	if(rank == 0)
	{
	    offset = stripSize + 1;
	    for(i=1; i<size; i++)
	    {
		MPI_Irecv(a[offset], stripSize*(n+2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &recv_request);
	        MPI_Wait(&recv_request,&status);
		offset += stripSize;
	    }
	}
	else
	{
	    MPI_Isend(a[rank*stripSize+1], stripSize*(n+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &send_request);
	    MPI_Wait(&send_request,&status);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Output final grid
	//printf("Final grid:\n");
	//print_grid(a,1,n+1);

	// Results
	if(rank == 0){
	printf("Results:\n");
	printf("Iterations=%d\n",iteration);
	printf("Tolerance=%12.10lf\n",maxdiff);
	printf("Running time=%12.8lf\n",ttotal);
	printf("Value at (%d,%d)=%12.8lf\n",r,c,a[r][c]);
	//printf("Value at (%d,%d)=%12.8lf\n",1,1,a[1][1]);
	//printf("Value at (%d,%d)=%12.8lf\n",1,4,a[1][4]);
	}
	/*
	if(rank == (size - 1))
	    printf("Value at (%d,%d)=%12.8lf\n",r,c,a[r][c]);
	*/
	//MPI Finalization
	MPI_Finalize();
	
	free_matrix(a,n+2);
	free_matrix(b,n+2);
	return 0;
}
