/*
This file contains functions needed for MPI runs of LATTICEEASY. Some involve communications between processors and others simply involve array manipulations on a single processor that are needed for the MPI version of the program.

The functions here which should be called externally are listed below.

The function cast_field_arrays(int array_size) allocates the memory for field and derivative arrays pointed to by f and fd. 1D: f[nflds][n+2], 2D: f[nflds][n+2][N+2], 3D: f[nflds][n+2][N][N+2].
The function free_field_arrays() frees the memory allocated for the arrays f and fd.
The function update_buffers() copies information from the edges of each processor's array to its neighboring processors' arrays.
*/

#include "latticeeasy.h"

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

// Allocate the memory for the field and derivative arrays
// The dimensions are 1D: f[nflds][n+2], 2D: f[nflds][n+2][N+2], 3D: f[nflds][n+2][N][N+2]. If FFTW requires a larger array, however, then the total array may include padding beyond these limits. However, the pointers allow you to ignore this and refer to the arrays as if they simply had the dimensions listed above.
// Note: This is done in a somewhat complicated way because of the tricky ways C++ deals with allocating multidimensional arrays. In 3D f[fld] is an array of pointers, each pointing to an array f[fld][n+2], which is an array of pointers each pointing to an array f[fld][n+2][N], which is an array of pointers each pointing to an array f[fld][n+2][N][N+2], which is finally an actual array of values. To ensure that all the values are stored in a contiguous block, however, the entire array is first allocated with the pointer fp, and then the arrays of pointers are set up to point to it as needed. The 1D and 2D cases are analogous.
void cast_field_arrays(int array_size)
{
#if NDIMS==1
  int fld;

  fp = new float[nflds*array_size]; // Cast an array of floats to hold f
  fdp = new float[nflds*array_size]; // Cast an array of floats to hold fdot

  // Allocate f and fd as arrays of pointers and then point them to the appropriate spots in the arrays of floats
  f = new float *[nflds];
  fd = new float *[nflds];
  for(fld=0;fld<nflds;fld++)
  {
    f[fld] = &(fp[fld*array_size]);
    fd[fld] = &(fdp[fld*array_size]);
  }
#elif NDIMS==2
  int fld, i;

  // The arrays are allocated to be big enough to hold the field [nflds*(n+2)*(N+2)], but they may be bigger if needed for the FFTs. The arrays of pointers allocated below will only be set up to point to the parts of the array needed for evolution.
  fp = new float[nflds*array_size]; // Cast an array of floats to hold f
  fdp = new float[nflds*array_size]; // Cast an array of floats to hold fdot

  // Allocate f and fd as arrays of pointers to pointers
  f = new float **[nflds];
  fd = new float **[nflds];
  for(fld=0;fld<nflds;fld++)
  {
    // For each value of fld allocate f[fld] and fd[fld] as arrays of pointers and then point them to the appropriate spots in the arrays of floats
    f[fld] = new float *[n+2];
    fd[fld] = new float *[n+2];
    for(i=0;i<n+2;i++)
    {
      f[fld][i] = &(fp[fld*array_size+i*(N+2)]);
      fd[fld][i] = &(fdp[fld*array_size+i*(N+2)]);
    }
  }
#elif NDIMS==3
  int fld, i, j;

  fp = new float[nflds*array_size]; // Cast an array of floats to hold f
  fdp = new float[nflds*array_size]; // Cast an array of floats to hold fdot

  // Allocate f and fd as arrays of pointers to pointers to pointers
  f = new float ***[nflds];
  fd = new float ***[nflds];
  for(fld=0;fld<nflds;fld++)
  {
    // For each value of fld allocate f[fld] and fd[fld] as arrays of pointers to pointers
    f[fld] = new float **[n+2];
    fd[fld] = new float **[n+2];
    for(i=0;i<n+2;i++)
    {
      // For each value of fld and i allocate f[fld][i] and fd[fld][i] as arrays of pointers and then point them to the appropriate spots in the arrays of floats
      f[fld][i] = new float *[N];
      fd[fld][i] = new float *[N];
      for(j=0;j<N;j++)
      {
	f[fld][i][j] = &(fp[fld*array_size+i*N*(N+2)+j*(N+2)]);
	fd[fld][i][j] = &(fdp[fld*array_size+i*N*(N+2)+j*(N+2)]);
      }
    }
  }
#endif

  // If grid images or full-array slices are being stored the root processor needs an extra array to gather data from other processors
  if(scheckpoint || (sslices && slicedim>=NDIMS))
  {
    MPI_Reduce(&n,&fstore_size,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD); // Find the largest value of n so that the array at the root processor will be big enough to hold values from any other processor
    if(my_rank==0)
    {
      if(NDIMS>1) fstore_size *= (N+2);
      if(NDIMS>2) fstore_size *= N;
      fstore = new float[fstore_size];
      if(fstore==NULL)
	printf("Unable to allocate array for slices and checkpointing. These functions will not be performed.\n");
    }
  }
}

// Delete the memory allocated for the field and derivative arrays
// See the note for cast_field_arrays() to see how the memory is structured. This function essentially reverses the steps in that one.
// (It might be sufficient to just call delete [] on f and fd, but I'm not sure.)
void free_field_arrays()
{
  // Free the memory used for the field values and derivatives
  delete [] fp;
  delete [] fdp;
  // Then free the memory used for arrays of pointers to those values
#if NDIMS==1
  delete [] f;
  delete [] fd;
#elif NDIMS==2
  int fld;

  for(fld=0;fld<nflds;fld++)
  {
    delete [] f[fld];
    delete [] fd[fld];
  }
  delete [] f;
  delete [] fd;
#elif NDIMS==3
  int fld, i;

  for(fld=0;fld<nflds;fld++)
  {
    for(i=0;i<n+2;i++)
    {
      delete [] f[fld][i];
      delete [] fd[fld][i];
    }
    delete [] f[fld];
    delete [] fd[fld];
  }
  delete [] f;
  delete [] fd;
#endif
}

#if NDIMS==1
#define FIELDADDRESS(fld,i) &(f[fld][i])
#elif NDIMS==2
#define FIELDADDRESS(fld,i) &(f[fld][i][0])
#elif NDIMS==3
#define FIELDADDRESS(fld,i) &(f[fld][i][0][0])
#endif
// Copy edge information between processors
void update_buffers()
{
  int fld, i=0, j=0, k=0;
  int edgesize=gridsize/N; // Size of each edge being sent
  int leftneighbor, rightneighbor; // Ranks of neighbor processors (accounting for periodicity of the lattice)
  MPI_Status status;

  if(NDIMS>1) // For NDIMS>1 the last index has extra padding (for FFTW to store the Nyquist frequencies in Fourier space)
    edgesize = edgesize*(N+2)/N;

  leftneighbor=(my_rank==0 ? numprocs-1 : my_rank-1); // leftneighbor is simply your rank minus one unless you're the zeroth processor, in which case it's the rightmost processor
  rightneighbor=(my_rank==numprocs-1 ? 0 : my_rank+1); // rightneighbor is simply your rank plus one unless you're the rightmost processor, in which case it's the zeroth processor

  for(fld=0;fld<nflds;fld++)
  {
    MPI_Sendrecv((void *) FIELDADDRESS(fld,n), edgesize, MPI_FLOAT, rightneighbor, my_rank,
		 (void *) FIELDADDRESS(fld,0), edgesize, MPI_FLOAT, leftneighbor, leftneighbor, MPI_COMM_WORLD, &status); // Send right edge and receive left edge
    MPI_Sendrecv((void *) FIELDADDRESS(fld,1), edgesize, MPI_FLOAT, leftneighbor, my_rank+numprocs,
		 (void *) FIELDADDRESS(fld,n+1), edgesize, MPI_FLOAT, rightneighbor, rightneighbor+numprocs, MPI_COMM_WORLD, &status); // Send left edge and receive right edge
  }

  buffers_up_to_date=1; // Note that buffers are now in synch
}
#undef FIELDADDRESS
