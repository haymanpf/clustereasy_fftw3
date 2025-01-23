/*
FFTEASY consists of the four C functions fftc1, fftcn, fftr1, and
fftrn. FFTEASY is free. I am not in any way, shape, or form expecting
to make money off of these routines. I wrote them because I needed
them for some work I was doing and I'm putting them out on the
Internet in case other people might find them useful. Feel free to
download them, incorporate them into your code, modify them, translate
the comment lines into Swahili, or whatever else you want. What I do
want is the following:
1) Leave this notice (i.e. this entire paragraph beginning with
``FFTEASY consists of...'' and ending with my email address) in with
the code wherever you put it. Even if you're just using it in-house in
your department, business, or wherever else I would like these credits
to remain with it. This is partly so that people can...
2) Give me feedback. Did FFTEASY work great for you and help your
work?  Did you hate it? Did you find a way to improve it, or translate
it into another programming language? Whatever the case might be, I
would love to hear about it. Please let me know at the email address
below.
3) Finally, insofar as I have the legal right to do so I forbid you
to make money off of this code without my consent. In other words if
you want to publish these functions in a book or bundle them into
commercial software or anything like that contact me about it
first. I'll probably say yes, but I would like to reserve that right.

This is the MPI version of FFTEASY, which consists of the four
functions listed above adapted for use in parallel computing.

For any comments or questions you can reach me at
gfelder@email.smith.edu.
*/

/*
The functions below are designed to be called by numprocs processors
simultaneously. Each processor should hold a portion of the total
array such that if the total array has dimensions [N1][N2][N3][...]
each processor's array has dimensions [n1][N2][N3][...]  where
n1=N1/numprocs. The array of Nyquist frequencies fnyquist[] is assumed
to be stored entirely at the root processor.
*/

/* These declarations are put here so you can easily cut and paste them into your program. */
void fftc1(float f[], int N, int skip, int forward, int my_rank, int numprocs);
void fftcn(float f[], int ndims, int size[], int forward, int my_rank, int numprocs);
void fftr1(float f[], int N, int forward, int my_rank, int numprocs);
void fftrn(float f[], float fnyquist[], int ndims, int size[], int forward, int my_rank, int numprocs);

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

struct complex
{
  float real;
  float imag;
};

/*
Do a Fourier transform of an array of N complex numbers separated by steps of (complex) size skip.
The array f should be of length 2N*skip (divided among numprocs processors) and N must be a power of 2.
Forward determines whether to do a forward transform (1) or an inverse one(-1)
*/
void fftc1(float f[], int N, int skip, int forward, int my_rank, int numprocs)
{
  int b,index1,index2,trans_size,trans,numtrans;
  float pi2 = 4.*asin(1.);
  float pi2n,cospi2n,sinpi2n; /* Used in recursive formulas for Re(W^b) and Im(W^b) */
  struct complex wb; /* wk = W^k = e^(2 pi i b/N) in the Danielson-Lanczos formula for a transform of length N */
  struct complex temp1,temp2; /* Buffers for implementing recursive formulas */
  struct complex *c = (struct complex *)f; /* Treat f as an array of N complex numbers */
  /* Variables needed for the MPI version */
  float *ftemp1,*ftemp2; /* Temporary storage array for values sent from other processors*/
  struct complex *ctemp1,*ctemp2;
  int indexlocal1,indexlocal2,proc1,proc2; /* index1 and index2 refer to indices on the entire array. indexlocal is for a particular processor's array. */ 
  int n=N/numprocs; /* The size of the local array at each processor */
  MPI_Status status; /* MPI: Return status for receive commmands */

  /* If there's more than one processor each one will need a temporary storage space for values sent by other processors */
  if(numprocs>1)
  {
    ftemp1 = (float *) malloc(2*n*sizeof(float));
    ftemp2 = (float *) malloc(2*n*sizeof(float));
    if(ftemp1==NULL || ftemp2==NULL)
    {
      printf("Failed to allocate memory in routine fftc1. Exiting.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
      exit(1);
    }
    ctemp1 = (struct complex *)ftemp1;
    ctemp2 = (struct complex *)ftemp2;
  }

  /* Place the elements of the array c in bit-reversed order */
  /* This is somewhat inefficient, but not too bad. I have each processor go through all indices and only use the ones that it has stored. */
  for(index1=1,index2=0;index1<N;index1++) /* Loop through all elements of c */
  {
    for(b=N/2;index2>=b;b/=2) /* To find the next bit reversed array index subtract leading 1's from index2 */
      index2-=b;
    index2+=b; /* Next replace the first 0 in index2 with a 1 and this gives the correct next value */
    if(index2>index1) /* Swap each pair only the first time it is found */
    {
      /* Find what processors have these two indices */
      proc1=index1/n;
      proc2=index2/n;
      if(proc1==my_rank && proc2==my_rank)
      {
	indexlocal1 = (index1%n)*skip;
	indexlocal2 = (index2%n)*skip;
	temp1 = c[indexlocal2];
	c[indexlocal2] = c[indexlocal1];
	c[indexlocal1] = temp1;
      }
      else if(proc1==my_rank)
      {
	indexlocal1 = (index1%n)*skip;
	MPI_Send((void *) &(c[indexlocal1]), 2, MPI_FLOAT, proc2, index1, MPI_COMM_WORLD);
	MPI_Recv((void *) &(c[indexlocal1]), 2, MPI_FLOAT, proc2, index2, MPI_COMM_WORLD, &status);
      }
      else if(proc2==my_rank)
      {
	indexlocal2 = (index2%n)*skip;
	MPI_Send((void *) &(c[indexlocal2]), 2, MPI_FLOAT, proc1, index2, MPI_COMM_WORLD);
	MPI_Recv((void *) &(c[indexlocal2]), 2, MPI_FLOAT, proc1, index1, MPI_COMM_WORLD, &status);	
      }
    } /* End of swap */
  }

  /* Next perform successive transforms of length 2,4,...,N using the Danielson-Lanczos formula */
  for(trans_size=2;trans_size<=N;trans_size*=2) /* trans_size = size of transform being computed */
  {
    numtrans=N/trans_size; /* The number of transforms of this size to be performed */
    pi2n = forward*pi2/(float)trans_size; /* +- 2 pi/trans_size */
    cospi2n = cos(pi2n); /* Used to calculate W^k in D-L formula */
    sinpi2n = sin(pi2n);
    if(numtrans>=numprocs) /* When there are as many transforms as processors each transform happens on a single processor */
    {
      wb.real = 1.; /* Initialize W^b for b=0 */
      wb.imag = 0.;
      for(b=0;b<trans_size/2;b++) /* Step over half of the elements in the transform */
      {
	for(trans=0;trans<n/trans_size;trans++) /* Iterate over all transforms of size trans_size to be computed */
	{
	  index1 = trans*trans_size+b; /* Index of element in first half of transform being computed */
	  index2 = index1 + trans_size/2; /* Index of element in second half of transform being computed */
	  /* Implement D-L formula */
	  indexlocal1 = index1*skip;
	  indexlocal2 = index2*skip;
	  temp1 = c[indexlocal1];
	  temp2 = c[indexlocal2];
	  c[indexlocal1].real = temp1.real + wb.real*temp2.real - wb.imag*temp2.imag;
	  c[indexlocal1].imag = temp1.imag + wb.real*temp2.imag + wb.imag*temp2.real;
	  c[indexlocal2].real = temp1.real - wb.real*temp2.real + wb.imag*temp2.imag;
	  c[indexlocal2].imag = temp1.imag - wb.real*temp2.imag - wb.imag*temp2.real;
	} /* End loop over trans */
	temp1 = wb;
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/trans_size) used in D-L formula */
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/trans_size) used in D-L formula */
      } /* End loop over b */
    } /* End if(numtrans>=numprocs) */
    else /* When there are more processors than processors each processor does only part of one transform */
    {
      trans = (n*my_rank)/trans_size; /* Find which transform this processor is working on */
      indexlocal1=(n*my_rank)%trans_size; /* Position of the start of this processor's array in the transform being computed*/ 
      /* Find what processor to exchange information with. */
      if(indexlocal1<trans_size/2) /* If you're in the first half of the transform it's above you */
	proc1 = my_rank + trans_size/2/n;
      else /* Otherwise it's below you */
	proc1 = my_rank - trans_size/2/n;
      for(b=0;b<n;b++) /* Copy all your values into a temporary array so you can send it without skips */
	ctemp1[b]=c[b*skip];
      MPI_Sendrecv((void *) ctemp1, 2*n, MPI_FLOAT, proc1, trans_size, (void *) ctemp2, 2*n, MPI_FLOAT, proc1, trans_size, MPI_COMM_WORLD, &status); /* Send a copy of your entire array and receive someone else's*/
      wb.real = cos(pi2n*(float)indexlocal1); /* Initialize W^b for the part of the transform this loop is going over */
      wb.imag = sin(pi2n*(float)indexlocal1);
      for(b=0;b<n;b++) /* Step over half of the elements in the transform */
      {
	/* Implement D-L formula */
	if(proc1>my_rank) /* The formula is different depending on whether you're in the first or second half of the transform */
	{
	  temp1 = ctemp1[b];
	  temp2 = ctemp2[b];
	}
	else
	{
	  temp2 = ctemp1[b];
	  temp1 = ctemp2[b];
	}
	c[b*skip].real = temp1.real + wb.real*temp2.real - wb.imag*temp2.imag;
	c[b*skip].imag = temp1.imag + wb.real*temp2.imag + wb.imag*temp2.real;
	/* Update wb coefficient */
	temp1 = wb;
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/trans_size) used in D-L formula */
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/trans_size) used in D-L formula */
      } /* End loop over b */
    } /* End (numtrans<numprocs) */
    fflush(stdout);
  } /* End loop over trans_size */

  /* For an inverse transform divide by the number of grid points */
  if(forward<0.)
    for(indexlocal1=0;indexlocal1<n*skip;indexlocal1+=skip)
    {
      c[indexlocal1].real /= N;
      c[indexlocal1].imag /= N;
    }

  /* Free memory allocated for temporary storage arrays */
  if(numprocs>1)
  {
    free(ftemp1);
    free(ftemp2);
  }
}

/* 
Do a Fourier transform of an ndims dimensional array of complex numbers
Array dimensions are given by size[0],...,size[ndims-1]. Note that these are sizes of complex arrays.
The array f should be of length 2*size[0]*...*size[ndims-1] and all sizes must be powers of 2.
Forward determines whether to do a forward transform (1) or an inverse one(-1)
*/
void fftcn(float f[], int ndims, int size[], int forward, int my_rank, int numprocs)
{
  int i,j,dim;
  int planesize=1,skip=1; /* These determine where to begin successive transforms and the skip between their elements (see below) */
  int totalsize=1; /* Total size of the ndims dimensional array */
  int index, proc, indexlocal; /* MPI: Index of the starting position in the total array, the rank of the processor holding that index, and the local index on that processor */
  int n, n0; /* MPI: Size of the local (complex) array at each processor, total and in the 0th dimension */
  float *ftemp; /* MPI: A temporary array for storing arrays on one processor for 1D FFTs */
  MPI_Datatype skipped_array; /* A new MPI datatype for passing complex elements at placed at regular intervals in an array */

  for(dim=0;dim<ndims;dim++) /* Determine total size of array */
    totalsize *= size[dim];
  n = totalsize/numprocs;
  n0 = size[0]/numprocs;

  /* Create a new MPI type for passing skipped data (see below) */
  MPI_Type_vector(n0,2,2*totalsize/size[0],MPI_FLOAT,&skipped_array);
  MPI_Type_commit(&skipped_array);

  if(my_rank==0)
  {
    ftemp = (float *) malloc(2*size[0]*sizeof(float)); /* 1D FFTs that span multiple processors will be passed to the root processor and done there for time efficiency */
    if(!ftemp)
    {
      printf("Failed to allocate memory in routine fftcn. Exiting.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
      exit(1);
    }
  }

  for(dim=ndims-1;dim>=0;dim--) /* Loop over dimensions */
  {
    planesize *= size[dim]; /* Planesize = Product of all sizes up to and including size[dim] */
    for(i=0;i<totalsize;i+=planesize) /* Take big steps to begin loops of transforms */
      for(j=0;j<skip;j++) /* Skip sets the number of transforms in between big steps as well as the skip between elements */
      {
	index = i+j;
	if(dim>0) /* For all but the highest array index the 1D array being transformed will be on a single processor */
	{
	  proc = index/n;
	  indexlocal = index%n;
	  if(proc==my_rank)
	    fftc1(f+2*indexlocal,size[dim],skip,forward,0,1); /* 1-D Fourier transform with numprocs=1. (Factor of two converts complex index to float index.) */
	}
	else
	{
	  indexlocal = index%n;
	  MPI_Gather((void *) &(f[2*indexlocal]), 1, skipped_array, (void *) ftemp, 2*n0, MPI_FLOAT, 0, MPI_COMM_WORLD); /* Copy all of the data for the 1D FFT onto the root processor */
	  if(my_rank==0)
	    fftc1(ftemp,size[0],1,forward,0,1); /* 1-D Fourier transform of the data that has been copied onto the root processor */
	  MPI_Scatter((void *) ftemp, 2*n0, MPI_FLOAT, (void *) &(f[2*indexlocal]), 1, skipped_array, 0, MPI_COMM_WORLD); /* Copy the data back to the other processors */
	}
      }
    skip *= size[dim]; /* Skip = Product of all sizes up to (but not including) size[dim] */
  }

  if(my_rank==0)
    free(ftemp); /* Free up memory allocated for temporary storage array */
}

/* 
Do a Fourier transform of an array of N real numbers
N must be a power of 2
Forward determines whether to do a forward transform (>=0) or an inverse one(<0)
*/
void fftr1(float f[], int N, int forward, int my_rank, int numprocs)
{
  int b;
  float pi2n = 4.*asin(1.)/N,cospi2n=cos(pi2n),sinpi2n=sin(pi2n); /* pi2n = 2 Pi/N */
  struct complex wb; /* wb = W^b = e^(2 pi i b/N) in the Danielson-Lanczos formula for a transform of length N */
  struct complex temp1,temp2; /* Buffers for implementing recursive formulas */
  struct complex *c = (struct complex *)f; /* Treat f as an array of N/2 complex numbers */
  /* Variables needed for the MPI version */
  int index1,index2,indexlocal1,indexlocal2,proc1,proc2; /* index1 and index2 refer to indices on the entire array. indexlocal is for a particular processor's array. */ 
  int n=N/numprocs/2; /* The size of the local array at each processor, measured in complex numbers (hence the 1/2) */
  MPI_Status status; /* MPI: Return status for receive commmands */

  if(forward==1)
    fftc1(f,N/2,1,1,my_rank,numprocs); /* Do a transform of f as if it were N/2 complex points */

  wb.real = 1.; /* Initialize W^b for b=0 */
  wb.imag = 0.;
  for(b=1;b<N/4;b++) /* Loop over elements of transform. See documentation for these formulas */
  {
    temp1 = wb;
    wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/N) used in D-L formula */
    wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/N) used in D-L formula */
    index1 = b; /* Complex index of first element */
    index2 = N/2-b; /* Complex index of second element */
    /* Find what processors have these two indices */
    proc1=index1/n;
    proc2=index2/n;
    if(proc1==my_rank && proc2==my_rank)
    {
      indexlocal1 = index1%n;
      indexlocal2 = index2%n;
      temp1 = c[indexlocal1];
      temp2 = c[indexlocal2];
      c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*wb.real*(temp1.imag+temp2.imag) + wb.imag*(temp1.real-temp2.real));
      c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
      c[indexlocal2].real = .5*(temp1.real+temp2.real - forward*wb.real*(temp1.imag+temp2.imag) - wb.imag*(temp1.real-temp2.real));
      c[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
    }
    else if(proc1==my_rank)
    {
      indexlocal1 = index1%n;
      temp1 = c[indexlocal1];
      MPI_Send((void *) &(c[indexlocal1]), 2, MPI_FLOAT, proc2, index1, MPI_COMM_WORLD);
      MPI_Recv((void *) &temp2, 2, MPI_FLOAT, proc2, index2, MPI_COMM_WORLD, &status);
      c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*wb.real*(temp1.imag+temp2.imag) + wb.imag*(temp1.real-temp2.real));
      c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
    }
    else if(proc2==my_rank)
    {
      indexlocal2 = index2%n;
      temp2 = c[indexlocal2];
      MPI_Send((void *) &(c[indexlocal2]), 2, MPI_FLOAT, proc1, index2, MPI_COMM_WORLD);
      MPI_Recv((void *) &temp1, 2, MPI_FLOAT, proc1, index1, MPI_COMM_WORLD, &status);	
      c[indexlocal2].real = .5*(temp1.real+temp2.real - forward*wb.real*(temp1.imag+temp2.imag) - wb.imag*(temp1.real-temp2.real));
      c[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
    }
  } /* End loop over b */
  if(my_rank==0) /* The lowest mode is handled separately, and is on the root processor. */
  {
    temp1 = c[0];
    c[0].real = temp1.real+temp1.imag; /* Set b=0 term in transform */
    c[0].imag = temp1.real-temp1.imag; /* Put b=N/2 term in imaginary part of first term */
  }

  if(forward==-1)
  {
    if(my_rank==0)
    {
      c[0].real *= .5;
      c[0].imag *= .5;
    }
    fftc1(f,N/2,1,-1,my_rank,numprocs);
  }
}

/* 
Do a Fourier transform of an ndims dimensional array of real numbers
Array dimensions are given by size[0],...,size[ndims-1]. All sizes must be powers of 2.
The (complex) nyquist frequency components are stored in fnyquist[size[0]][size[1]]...[2*size[ndims-2]]
Forward determines whether to do a forward transform (1) or an inverse one(-1)
*/
void fftrn(float f[], float fnyquist[], int ndims, int size[], int forward, int my_rank, int numprocs)
{
  int i,j,b;
  int index,indexneg=0; /* Positions in the 1-d arrays of points labeled by indices (i0,i1,...,i(ndims-1)); indexneg gives the position in the array of the corresponding negative frequency */
  int stepsize; /* Used in calculating indexneg */
  int N=size[ndims-1]; /* The size of the last dimension is used often enough to merit its own name. */
  double pi2n = 4.*asin(1.)/N,cospi2n=cos(pi2n),sinpi2n=sin(pi2n); /* pi2n = 2 Pi/N */
  struct complex wb; /* wb = W^b = e^(2 pi i b/N) in the Danielson-Lanczos formula for a transform of length N */
  struct complex temp1,temp2; /* Buffers for implementing recursive formulas */
  struct complex *c = (struct complex *)f, *cnyquist = (struct complex *)fnyquist; /* Treat f and fnyquist as arrays of complex numbers */
  int totalsize=1; /* Total number of complex points in array */
  int *indices= (int *) malloc(ndims*sizeof(int)); /* Indices for looping through array */
  /* Variables needed for the MPI version */
  int index1,index2,indexlocal1,indexlocal2,proc1,proc2; /* index1 and index2 refer to indices on the entire array. indexlocal is for a particular processor's array. */ 
  int n; /* The size of the local array at each processor */
  MPI_Status status; /* MPI: Return status for receive commmands */
  struct complex *ctemp = (struct complex *) malloc(N/2*sizeof(float)); /* An array for temporary storage of variables from other processors */

  if(!indices || !ctemp) /* Make sure memory was correctly allocated */
  {
    printf("Error allocating memory in fftrn routine. Exiting.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
    exit(1);
  }

  size[ndims-1] /= 2; /* Set size[] to be the sizes of f viewed as a complex array */
  for(i=0;i<ndims;i++)
  {
    totalsize *= size[i];
    indices[i] = 0;
    n = totalsize/numprocs;
  }

  if(forward==1) /* Forward transform */
  {
    fftcn(f,ndims,size,1,my_rank,numprocs); /* Do a transform of f as if it were N/2 complex points */
    for(i=0;i<totalsize/size[ndims-1];i++) /* Copy b=0 data into cnyquist so the recursion formulas below for b=0 and cnyquist don't overwrite data they later need */
    {
      index1 = i*size[ndims-1]; /* Complex index of element in the entire array. Only copy points where last array index for c is 0. */
      proc1=index1/n; /* Find what processors has this index */
      if(proc1==my_rank)
      {
	indexlocal1 = index1%n; /* Complex index of element in the local array */
	MPI_Send((void *) &(c[indexlocal1]), 2, MPI_FLOAT, 0, index1, MPI_COMM_WORLD);
      }
      if(my_rank==0)
	MPI_Recv((void *) &(cnyquist[i]), 2, MPI_FLOAT, proc1, index1, MPI_COMM_WORLD, &status);	
    }
  }

  for(index=0;index<totalsize;index+=size[ndims-1]) /* Loop over all but last array index */
  {
    wb.real = 1.; /* Initialize W^b for b=0 */
    wb.imag = 0.;
    /* Find what processors have these two indices */
    proc1=index/n;
    proc2=indexneg/n;
    if(proc1==my_rank && proc2==my_rank)
    {
      for(b=1;b<N/4;b++) /* Loop over elements of transform. See documentation for these formulas */
      {
	temp1 = wb;
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/N_real) used in D-L formula */
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/N_real) used in D-L formula */
	index1 = index+b; /* Complex index of first element */
	index2 = indexneg+N/2-b; /* Complex index of second element. Note that N-b is NOT the negative frequency for b. Only nonnegative b momenta are stored. */
	indexlocal1 = index1%n;
	indexlocal2 = index2%n;
	temp1 = c[indexlocal1];
	temp2 = c[indexlocal2];
	c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*wb.real*(temp1.imag+temp2.imag) + wb.imag*(temp1.real-temp2.real));
	c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
	c[indexlocal2].real = .5*(temp1.real+temp2.real - forward*wb.real*(temp1.imag+temp2.imag) - wb.imag*(temp1.real-temp2.real));
	c[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
      } /* End loop over b */
    } /* End if both processors are the same */
    else if(proc1==my_rank)
    {
      MPI_Sendrecv((void *) &(c[index%n]), N/2, MPI_FLOAT, proc2, index, (void *) ctemp, N/2, MPI_FLOAT, proc2, indexneg, MPI_COMM_WORLD, &status); /* Send and receive a copy of the array being altered */
      for(b=1;b<N/4;b++) /* Loop over elements of transform. See documentation for these formulas */
      {
	temp1 = wb;
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/N_real) used in D-L formula */
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/N_real) used in D-L formula */
	index1 = index+b; /* Complex index of first element */
	index2 = N/4-b; /* Complex index of second element. Note that N-b is NOT the negative frequency for b. Only nonnegative b momenta are stored. */
	indexlocal1 = index1%n;
	temp1 = c[indexlocal1];
	temp2 = ctemp[index2];
	c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*wb.real*(temp1.imag+temp2.imag) + wb.imag*(temp1.real-temp2.real));
	c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
      }
    } /* End if(proc1==my_rank) */
    else if(proc2==my_rank)
    {
      MPI_Sendrecv((void *) &(c[indexneg%n+N/4]), N/2, MPI_FLOAT, proc1, indexneg, (void *) ctemp, N/2, MPI_FLOAT, proc1, index, MPI_COMM_WORLD, &status); /* Send and receive a copy of the array being altered */
      for(b=1;b<N/4;b++) /* Loop over elements of transform. See documentation for these formulas */
      {
	temp1 = wb;
	wb.real = cospi2n*temp1.real - sinpi2n*temp1.imag; /* Real part of e^(2 pi i b/N_real) used in D-L formula */
	wb.imag = cospi2n*temp1.imag + sinpi2n*temp1.real; /* Imaginary part of e^(2 pi i b/N_real) used in D-L formula */
	index1 = b; /* Complex index of first element */
	index2 = indexneg+N/2-b; /* Complex index of second element. Note that N-b is NOT the negative frequency for b. Only nonnegative b momenta are stored. */
	indexlocal2 = index2%n;
	temp1 = ctemp[index1];
	temp2 = c[indexlocal2];
	c[indexlocal2].real = .5*(temp1.real+temp2.real - forward*wb.real*(temp1.imag+temp2.imag) - wb.imag*(temp1.real-temp2.real));
	c[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*wb.real*(temp1.real-temp2.real) + wb.imag*(temp1.imag+temp2.imag));
      }
    } /* End if(proc2==my_rank) */
    /* For now at least the handling of the k=0 and Nyquist frequencies are very inefficient. For each value of the other indices a message is sent back and forth to root, where cnyquist is stored */
    index1 = index; /* Complex index of first element */
    index2 = indexneg/size[ndims-1]; /* Complex index of second element. Index is smaller for cnyquist because it doesn't have the last dimension. */
    /* Find what processors have these two indices */
    proc1=index1/n;
    proc2=0; /* The Nyquist array is all at the root processor. */
    if(proc1==my_rank && proc2==my_rank)
    {
      indexlocal1 = index1%n;
      indexlocal2 = index2;
      temp1 = c[indexlocal1];
      temp2 = cnyquist[indexlocal2];
      /* Set b=0 term in transform */
      c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*(temp1.imag+temp2.imag));
      c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*(temp1.real-temp2.real));
      /* Set b=N/2 transform. */
      cnyquist[indexlocal2].real = .5*(temp1.real+temp2.real - forward*(temp1.imag+temp2.imag));
      cnyquist[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*(temp1.real-temp2.real));
    }
    else if(proc1==my_rank)
    {
      indexlocal1 = index1%n;
      temp1 = c[indexlocal1];
      MPI_Send((void *) &(c[indexlocal1]), 2, MPI_FLOAT, proc2, index1, MPI_COMM_WORLD);
      MPI_Recv((void *) &temp2, 2, MPI_FLOAT, proc2, index2, MPI_COMM_WORLD, &status);
      /* Set b=0 term in transform */
      c[indexlocal1].real = .5*(temp1.real+temp2.real + forward*(temp1.imag+temp2.imag));
      c[indexlocal1].imag = .5*(temp1.imag-temp2.imag - forward*(temp1.real-temp2.real));
    }
    else if(proc2==my_rank)
    {
      indexlocal2 = index2;
      temp2 = cnyquist[indexlocal2];
      MPI_Send((void *) &(cnyquist[indexlocal2]), 2, MPI_FLOAT, proc1, index2, MPI_COMM_WORLD);
      MPI_Recv((void *) &temp1, 2, MPI_FLOAT, proc1, index1, MPI_COMM_WORLD, &status);	
      /* Set b=N/2 transform. */
      cnyquist[indexlocal2].real = .5*(temp1.real+temp2.real - forward*(temp1.imag+temp2.imag));
      cnyquist[indexlocal2].imag = .5*(-temp1.imag+temp2.imag - forward*(temp1.real-temp2.real));
    }
    
    /* Find indices for positive and single index for negative frequency. In each dimension indexneg[j]=0 if index[j]=0, indexneg[j]=size[j]-index[j] otherwise. */
    stepsize=size[ndims-1]; /* Amount to increment indexneg by as each individual index is incremented */
    for(j=ndims-2;indices[j]==size[j]-1 && j>=0;j--) /* If the rightmost indices are maximal reset them to 0. Indexneg goes from 1 to 0 in these dimensions. */
    {
      indices[j]=0;
      indexneg -= stepsize;
      stepsize *= size[j];
    }
    if(indices[j]==0) /* If index[j] goes from 0 to 1 indexneg[j] goes from 0 to size[j]-1 */
      indexneg += stepsize*(size[j]-1);
    else /* Otherwise increasing index[j] decreases indexneg by one unit. */
      indexneg -= stepsize;
    if(j>=0) /* This avoids writing outside the array bounds on the last pass through the array loop */
      indices[j]++;
  } /* End of i loop (over total array) */


  if(forward==-1) /* Inverse transform */
    fftcn(f,ndims,size,-1,my_rank,numprocs);

  size[ndims-1] *= 2; /* Give the user back the array size[] in its original condition */
  /* Free up memory allocated for arrays */
  free(indices);
  free(ctemp);
}

