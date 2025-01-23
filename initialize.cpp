/*
This file contains the initialization function for setting field and parameter values.

The only function here that should be called externally is initialize(). It sets field values in momentum space and then transforms them to position space. It also sets parameters such as the scale factor and its derivatives.

MPI version: The initialize() function also calls the function that casts the field arrays (and sets all the needed pointers to them).
*/

#include "latticeeasy.h"

// Generate a uniform deviate between 0 and 1 using the Park-Miller minimum standard algorithm
#define randa 16807
#define randm 2147483647
#define randq 127773
#define randr 2836
float rand_uniform(void)
{
  if(seed<1) return(0.33); // *DEBUG* This is used to avoid randomness, for debugging purposes only.
  static int i=0;
  static int next=seed+my_rank;
  if(!(next>0)) // Guard against 0, negative, or other invalid seeds
    {
      printf("Invalid seed used in random number function. Using seed=1\n");
      next=1;
    }
  if(i==0) // On the first call run through 100 calls. This allows small seeds without the first results necessarily being small.
    for(i=1;i<100;i++)
      rand_uniform();
  next = randa*(next%randq) - randr*(next/randq);
  if(next<0) next += randm;
  return ((float)next/(float)randm);
}
#undef randa
#undef randm
#undef randq
#undef randr

// Set the amplitude and phase of a mode of vacuum fluctuations
// Phase is set randomly (except for modes that must be real).
// Amplitude is set with a Rayleigh distribution about an rms value of 1/sqrt(omega) (times normalization terms).
void set_mode(float p2, float m2, float *field, float *deriv, int real)
{
  float phase,amplitude,rms_amplitude,omega;
  float re_f_left,im_f_left,re_f_right,im_f_right;
#if NDIMS==1
  static float norm = rescale_A*rescale_B*pow(L/pw2(dx),.5)/sqrt(4.*pi);
#elif NDIMS==2
  static float norm = rescale_A*rescale_B*L/pw2(dx)/sqrt(2.*pi);
#elif NDIMS==3
  static float norm = rescale_A*rescale_B*pow(L/pw2(dx),1.5)/sqrt(2.);
#endif
  static float hbterm = hubble_init*(rescale_r-1.);
  static int tachyonic=0; // Avoid printing the same error repeatedly

  // Momentum cutoff. If kcutoff!=0 then eliminate all initial modes with k>kcutoff.
  static float k2cutoff = (kcutoff<2.*pi*(float)N/L ? pw2(kcutoff) : 0.);
  if(k2cutoff>0. && p2>k2cutoff)
  {
    field[0]=0.;
    field[1]=0.;
    deriv[0]=0.;
    deriv[1]=0.;
    return;
  }

  if(p2+m2>0.) // Check to avoid floating point errors
    omega=sqrt(p2+m2); // Omega = Sqrt(p^2 + m^2)
  else
  {
    if(tachyonic==0)
      printf("Warning: Tachyonic mode(s) may be initialized inaccurately\n");
    omega=sqrt(p2); // If p^2 + m^2 < 0 use m^2=0
    tachyonic=1;
  }
  if(omega>0.) // Avoid dividing by zero
    rms_amplitude=norm/sqrt(omega)*pow(p2,.75-(float)NDIMS/4.);
  else
    rms_amplitude=0.;

  // Amplitude = RMS amplitude x Rayleigh distributed random number
  // The same amplitude is used for left and right moving waves to generate standing waves. The extra 1/sqrt(2) normalizes the initial occupation number correctly.
  amplitude=rms_amplitude/sqrt(2.)*sqrt(log(1./rand_uniform()));
  // Left moving component
  phase=2.*pi*rand_uniform(); // Set phase randomly
  re_f_left = amplitude*cos(phase);
  im_f_left = amplitude*sin(phase);
  // Right moving component
  phase=2.*pi*rand_uniform(); // Set phase randomly
  re_f_right = amplitude*cos(phase);
  im_f_right = amplitude*sin(phase);

  field[0] = re_f_left + re_f_right; // Re(field)
  field[1] = im_f_left + im_f_right; // Im(field)
  deriv[0] = omega*(im_f_left - im_f_right) + hbterm*field[0]; // Field derivative
  deriv[1] = -omega*(re_f_left - re_f_right) + hbterm*field[1];
  if(real==1) // For real modes set the imaginary parts to zero
  {
    field[1] = 0.;
    deriv[1] = 0.;
  }

  return ;
}

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

// Set initial parameters and field values
void initialize()
{
  int fld;
  float p2; // Total squared momentum
  float dp2=pw2(2.*pi/L); // Square of grid spacing in momentum space
  float mass_sq[nflds]; // Effective mass squared of fields
  float initial_field_values[nflds]; // Initial value of fields (set to zero unless specified in parameters.h)
  float initial_field_derivs[nflds]; // Initial value of field derivatives (set to zero unless specified in parameters.h)
  float fsquared=0.,fdsquared=0.,ffd=0.,pot_energy=0.; // Sum(field^2), Sum(field_dot^2), Sum(f*f_dot), and potential energy - used for calculating the Hubble constant
  FILE *old_grid_; // Used for reading in previously generated data as initial conditions
  // Variables needed for the MPI version of LATTICEEASY
  MPI_Status status; // MPI: Return status for receive commmands
  int proc,conjrank; // MPI: Processor indices
  int fieldstart; // Used for marking a position in the grid file when that file is used to read in data from a previous run
#if NDIMS==1
  int i,k;
  int pz; // Component of momentum in units of grid spacing
  n=N/numprocs; // Divide the grid evenly among the processors. (This is required by FFTEASY.)
  my_start_position=my_rank*n; // Each processor starts at evenly spaced positions in the array
  if(n*numprocs!=N) // Make sure that the grid divided evenly. (This won't necessarily work because of the way integer arithmetic works.)
  {
    printf("The number of processors must be a divisor of N. Exiting.\n");
    MPI_Finalize();
    exit(1);
  }
  cast_field_arrays(n+2); // Cast the arrays for the fields and their derivatives. The argument is the total size per field of the local array.
#elif NDIMS==2
  int i,j,k,jconj; // jconj is the index where conjugate modes to j are stored
  int py,pz; // Components of momentum in units of grid spacing
  float *fnyquist,*fdnyquist; // Used for modes with k=nyquist. These are calculated on the root processor but then they are sent to the other processors for the FFT.
  MPI_Datatype skipped_array; // A new MPI datatype for passing complex elements at placed at regular intervals in an array. This is used for passing the Nyquist frequencies.
  // rfftwnd_mpi_plan plan; // Plan for calculating FFT
  fftwf_plan plans_f[nflds]; // Plans for calculating inverse FFT. In FFTW3, plans are specific to individual data sets, so we need one for each field and each field derivative. Note we only need inverse for init. // PH
  fftwf_plan plans_fd[nflds]; // Plans for calculating inverse FFT. In FFTW3, plans are specific to individual data sets, so we need one for each field and each field derivative. Note we only need inverse for init. // PH
    
  ptrdiff_t local_nj, local_j_start, total_local_size; // Parameters of the local lattice given by FFTW // PH --- Docs use ptrdiff, but I'm going to try just ints for now.
  int tag; // Used for MPI messages
  int array_size; // Total size of the local field array (per field)
  int *all_n; // Array for storing the value of n at all processors. The root processor needs to know this to distribute information correctly

  total_local_size = fftwf_mpi_local_size_2d(N, N/2+1, MPI_COMM_WORLD, &local_nj, &local_j_start); // PH
  total_local_size *= 2.; // PH --- FFTW3 calculates complex N instead of real. This is a change from FFTW2.

  n=local_nj; // Set the global variable n to the correct value for this grid's processor
  my_start_position = local_j_start; // Set the global variable my_start_position for this processor
  array_size = ( (n+2)*(N+2) > total_local_size+N+2 ? (n+2)*(N+2) : total_local_size+N+2 ); // See the documentation for details on the needed array size
  cast_field_arrays(array_size); // Cast the arrays for the fields and their derivatives. The argument is the total size per field of the local array.

  for(fld=0;fld<nflds;fld++) // Initialize the plans for each field. Note the docs say this is roughly as fast as setting a single plan since all parameters are the same. // PH 
  {
    plans_f[fld]  = fftwf_mpi_plan_dft_c2r_2d(N, N, (fftwf_complex *)&(f[fld][1][0]), &(f[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);
    plans_fd[fld] = fftwf_mpi_plan_dft_c2r_2d(N, N, (fftwf_complex *)&(fd[fld][1][0]), &(fd[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);
  }

  if(my_rank==0) // Cast the arrays for storing the Nyquist arrays at the root processor and get the sizes of all local grids for distributing those modes.
  {
    fnyquist = new float[2*N];
    fdnyquist = new float[2*N];
    all_n = new int[numprocs];
    all_n[0]=n;
    for(proc=1;proc<numprocs;proc++)
      MPI_Recv((void *)(all_n+proc), 1, MPI_INT, proc, proc, MPI_COMM_WORLD, &status);
  }
  else
    MPI_Send((void *)&n,1,MPI_INT,0,my_rank,MPI_COMM_WORLD);
  MPI_Type_vector(n, 2, (N+2), MPI_FLOAT, &skipped_array); // Define skipped_array as having n blocks of length 2 separated by a skip of N+2
  MPI_Type_commit(&skipped_array); // Commit the datatype. (I have no idea why MPI requires this separate step after defining a datatype.)
#elif NDIMS==3
  int i,j,k,iconj,jconj; // iconj and jconj are the indices where conjugate modes to i and j are stored
  int px,py,pz; // Components of momentum in units of grid spacing
  float (*fnyquist)[2*N],(*fdnyquist)[2*N]; // Used for modes with k=nyquist. These are calculated on the root processor but then they are sent to the other processors for the FFT.
  MPI_Datatype skipped_array; // A new MPI datatype for passing complex elements at placed at regular intervals in an array. This is used for passing the Nyquist frequencies.
  fftwf_plan plans_f[nflds]; // Plan for calculating FFT // PH
  fftwf_plan plans_fd[nflds]; // Plan for calculating FFT // PH
  ptrdiff_t local_nx, local_i_start, total_local_size; // Parameters of the local lattice given by FFTW // PH
  int *all_n; // Array for storing the value of n at all processors. The root processor needs to know this to distribute information correctly
  int tag; // Used for MPI messages

  total_local_size = fftwf_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &local_nx, &local_i_start); // PH
  total_local_size *= 2.; // PH --- FFTW3 calculates complex N instead of real. This is a change from FFTW2.
  n=local_nx; // Set the global variable n to the correct value for this grid's processor
  my_start_position = local_i_start; // Set the global variable my_start_position for this processor
  cast_field_arrays((n+2)*N*(N+2)); // Cast the field arrays

  for(fld=0;fld<nflds;fld++) // Initialize the plans for each field. Note the docs say this is roughly as fast as setting a single plan since all parameters are the same. // PH 
  {
    plans_f[fld]  = fftwf_mpi_plan_dft_c2r_3d(N, N, N, (fftwf_complex *)&(f[fld][1][0][0]), &(f[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);
    plans_fd[fld] = fftwf_mpi_plan_dft_c2r_3d(N, N, N, (fftwf_complex *)&(fd[fld][1][0][0]), &(fd[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);
  }

  if(my_rank==0) // Cast the arrays for storing the Nyquist arrays at the root processor and get the sizes of all local grids for distributing those modes.
  {   
    fnyquist = new float[N][2*N];
    fdnyquist = new float[N][2*N];
    all_n = new int[numprocs];
    all_n[0]=n;
    for(proc=1;proc<numprocs;proc++)
      MPI_Recv((void *)(all_n+proc), 1, MPI_INT, proc, proc, MPI_COMM_WORLD, &status);
  }
  else
    MPI_Send((void *)&n,1,MPI_INT,0,my_rank,MPI_COMM_WORLD);
  MPI_Type_vector(n*N, 2, (N+2), MPI_FLOAT, &skipped_array); // Define skipped_array as having n*N blocks of length 2 separated by a skip of N+2
  MPI_Type_commit(&skipped_array); // Commit the datatype. (I have no idea why MPI requires this separate step after defining a datatype.)
#endif
  // Check to make sure time step is small enough to satisfy Courant condition, dt/dx < 1/Sqrt(ndims)
  if(dt>dx/sqrt((double)NDIMS))
  {
    if(my_rank==0)
    {
      printf("Time step too large. The ratio dt/dx is currently %f but for stability should never exceed 1/sqrt(%d) (%f)\n",dt/dx,NDIMS,1./sqrt((double)NDIMS));
      printf("Adjust dt to AT MOST %e, and preferably somewhat smaller than that.\n",dx/sqrt((double)NDIMS));
    }
    MPI_Finalize();
    exit(1);
  }

  modelinitialize(1); // This allows specific models to perform any needed initialization

  // Output initializations - Set values of nfldsout and ext_
  nfldsout = (noutput_flds==0 || noutput_flds>nflds ? nflds : noutput_flds); // Set number of files to output
  if(alt_extension[0]!='\0') // If an alternate extension was given use that instead of the default "_<run_number>.dat"
    sprintf(ext_,"%s",alt_extension);

  if(continue_run>0 && (old_grid_=fopen("grid.img","rb"))) // If an old grid image can be opened use it to read in initial conditions
  {
    if(my_rank==0)
      printf("Previously generated grid image found. Reading in data...\n");
    // Read in general data (time, scale factor, etc.)
    fread(&run_number,sizeof(run_number),1,old_grid_);
    run_number++;
    fread(&t0,sizeof(t0),1,old_grid_);
    if(t0>=tf) // Check to make sure that the time in the old grid is earlier than the final time for this run
    {
      if(my_rank==0)    
	printf("A grid image file was found in this directory with values stored at t=%f. To continue that run set tf to a later time. To start a new run move or rename the file grid.img.\n",t0);
      MPI_Finalize();
      exit(1);
    }
    fread(&a,sizeof(a),1,old_grid_);
    fread(&ad,sizeof(ad),1,old_grid_);
    fieldstart = ftell(old_grid_); // Record the position of the stream at the start of the field data

    // Read in field and derivative values
    for(fld=0;fld<nflds;fld++)
    {
#if NDIMS==1
      fseek(old_grid_,fieldstart+sizeof(float)*(fld*N+my_start_position),SEEK_SET); // Set file position to the start of this processor's data
      fread(&(f[fld][1]),sizeof(float),n,old_grid_); // Read in the field data
      fseek(old_grid_,fieldstart+sizeof(float)*((fld+nflds)*N+my_start_position),SEEK_SET); // Set file position to the start of this processor's data
      fread(&(fd[fld][1]),sizeof(float),n,old_grid_); // Read in the field derivative data
#elif NDIMS==2
      fseek(old_grid_,fieldstart+sizeof(float)*(fld*N*N+my_start_position*N),SEEK_SET); // Set file position to the start of this processor's data
      // Read in field values
      for(i=1;i<=n;i++) // Read each row (i value) separately to skip the padding values in the second dimension
	fread(&(f[fld][i][0]),sizeof(float),N,old_grid_); // Read in the field data
      fseek(old_grid_,fieldstart+sizeof(float)*((fld+nflds)*N*N+my_start_position*N),SEEK_SET); // Set file position to the start of this processor's data
      // Read in field derivative values
      for(i=1;i<=n;i++) // Read each row (i value) separately to skip the padding values in the second dimension
	fread(&(fd[fld][i][0]),sizeof(float),N,old_grid_); // Read in the field data
#elif NDIMS==3
      fseek(old_grid_,fieldstart+sizeof(float)*(fld*N*N*N+my_start_position*N*N),SEEK_SET); // Set file position to the start of this processor's data
      // Read in field values
      for(i=1;i<=n;i++) // Read each row (i and j values) separately to skip the padding values in the second dimension
	for(j=0;j<N;j++)
	  fread(&(f[fld][i][j][0]),sizeof(float),N,old_grid_); // Read in the field data
      fseek(old_grid_,fieldstart+sizeof(float)*((fld+nflds)*N*N*N+my_start_position*N*N),SEEK_SET); // Set file position to the start of this processor's data
      // Read in field derivative values
      for(i=1;i<=n;i++) // Read each row (i and j values) separately to skip the padding values in the second dimension
	for(j=0;j<N;j++)
	  fread(&(fd[fld][i][j][0]),sizeof(float),N,old_grid_); // Read in the field data
#endif
    } // End of loop over fields
    fclose(old_grid_);
    if(continue_run==1) // Option to append new data to old data files
      sprintf(mode_,"a+");
    else if(alt_extension[0]=='\0') // If no alternate extension was given set filename extension to indicate run number
      sprintf(ext_,"_%d.dat",run_number);
    buffers_up_to_date=0; // Buffers have now been set out of synch and will need to be updated before any gradients are calculated
    if(my_rank==0)
    {
      printf("Data read. Resuming run at t=%f\n",t0);
      output_parameters(); // Save information about the model and parameters and start the clock for timing the run
    }
    return;
  } // End of reading old grid images

  // If the variable no_initialization is set to 1 by the model file then don't initialize the modes
  if(no_initialization==1)
  {
    save(0); // Save field values and derived quantities
    if(my_rank==0)
      output_parameters(); // Save information about the model and parameters and start the clock for timing the run
    return;
  }

  // If no old grid image is found generate new initial conditions and set run_number=0
  if(my_rank==0)
    printf("Generating initial conditions for new run at t=0\n");
  t0=0;
  run_number=0;

  for(fld=0;fld<nflds;fld++)
  {
    // Set initial field values
    if(fld<(int)(sizeof initfield/sizeof(float))) // Use preset values for initial field values if given
      initial_field_values[fld]=initfield[fld];
    else // Otherwise initialize field to zero
      initial_field_values[fld]=0.;

    // Set initial field derivative values
    if(fld<(int)(sizeof initderivs/sizeof(float))) // Use preset values for initial field derivative values if given
      initial_field_derivs[fld]=initderivs[fld];
    else // Otherwise initialize derivatives to zero
      initial_field_derivs[fld]=0.;
  }

  // Set initial values of effective mass.
  effective_mass(mass_sq,initial_field_values);

  // Set initial value of Hubble constant - See the documentation for an explanation of the formulas used
  if(expansion>0)
  {
    if(expansion==1 && my_rank==0)
      printf("The initial value of the fields is being used to determine the initial Hubble constant.\nFrom here on power law expansion will be used\n");
    for(fld=0;fld<nflds;fld++) // Find sum of squares of fields and derivatives
    {
      fsquared += pw2(initial_field_values[fld]);
      fdsquared += pw2(initial_field_derivs[fld]);
      ffd += initial_field_values[fld]*initial_field_derivs[fld];
    }
    for(i=0;i<num_potential_terms;i++) // Find potential energy
      pot_energy += potential_energy(i,initial_field_values);
    hubble_init = sqrt( 3.*pw2(rescale_A)/(4.*pi)*fdsquared + 2.*pot_energy*(3.*pw2(rescale_A)/(4.*pi)-pw2(rescale_r)*fsquared) );
    hubble_init -= rescale_r*ffd;
    hubble_init /= 3.*pw2(rescale_A)/(4.*pi)-pw2(rescale_r)*fsquared;
    if(!(hubble_init>=0.)) // Make sure Hubble isn't negative or undefined
    {
      if(my_rank==0)
	printf("Error in calculating initial Hubble constant. Exiting.\n");
      MPI_Finalize();
      exit(1);
    }
    ad=hubble_init;
  }

  for(fld=0;fld<nflds;fld++) // Set initial conditions for each field
  {
#if NDIMS==1
    if(my_rank==0) // The modes k=0 and k=N/2 are stored in the first two positions on the root processor.
    {
      // Set zeromode of field and derivative to zero (it gets added in position space)
      f[fld][1]=0.; // (Remember that these arrays are unit-offset in the first position index.)
      fd[fld][1]=0.;
      // Set mode with k=N/2. This is a real number stored in the position f[fld][1].
      // This is done before the loop because this would overwrite the real part of k=1 otherwise.
      p2=dp2*pw2(N/2); // Total momentum squared for k=N/2
      set_mode(p2,mass_sq[fld],&f[fld][2],&fd[fld][2],1); // The last argument specifies that a real value should be set
    }

    // Loop over gridpoints.
    for(k=0;k<n/2;k++) // k is a complex index on the local array
    {
      if(my_rank==0 && k==0) continue; // Don't redo the zeromode
      pz=my_rank*n/2+k; // z-component of momentum of modes at z=k
      p2=dp2*pw2(pz); // Total momentum squared
      set_mode(p2,mass_sq[fld],&f[fld][2*k+1],&fd[fld][2*k+1],0); // Set mode
    }

    // *DEBUG* The option to not use the FFTs is for debugging purposes only. For actual simulations seed should always be positive
    if(seed>=0)
    {
      fftr1(&(f[fld][1]),N,-1,my_rank,numprocs); // Inverse Fourier transform of field
      fftr1(&(fd[fld][1]),N,-1,my_rank,numprocs); // Inverse Fourier transform of field derivatives
    }
#elif NDIMS==2
    for(j=1; j<=n; j++)
    {
      py = j+local_j_start-1; // y index in the entire array (not just this processor's array)
      py = (py<=N/2 ? py : py-N); // The momentum in wrap around order. 
      for(k=1; k<N/2; k++) // Modes in the bulk (k>0, k<N/2) are not real and have no complex conjugate mode in the lattice
      {
	pz = k;
	p2=dp2*(pw2(py)+pw2(pz));
	set_mode(p2,mass_sq[fld],&f[fld][j][2*k],&fd[fld][j][2*k],0); // Set mode.
      } // End loop over k
    } // End loop over j

    // Modes with k=0 or k=N/2 are calculated in the nyquist array on the root processor and then copied to the field arrays of the appropriate processors.
    for(k=0; k<=N/2; k+=N/2) // Edges of k contain columns of conjugated values
    {
      pz=k;
      if(my_rank==0) // Initialize all conjugated/real values on the root processor.
      {
	// First set all of the correct values in the nyquist array
	for(j=N/2+1; j<N; j++) // The lower "half" will be initialized as the conjugate of the upper half
	{
	  py=j-N;
	  jconj=-py; // The location of the mode conjugate to the mode at location py
	  p2=dp2*(pw2(py)+pw2(pz));
	  set_mode(p2, mass_sq[fld], &fnyquist[2*j], &fdnyquist[2*j], 0); // Set the mode in the "upper half"
	  // Next set the corresponding "lower half" mode to the complex conjugate of the "upper half" mode
	  fnyquist[2*jconj] = fnyquist[2*j];
	  fnyquist[2*jconj+1] = -fnyquist[2*j+1];
	  fdnyquist[2*jconj] = fdnyquist[2*j];
	  fdnyquist[2*jconj+1] = -fdnyquist[2*j+1];
	}
	for(j=0; j<=N/2; j+=N/2) // The points with py and pz each equal to 0 or N/2 are real
	{
	  py=j;
	  p2=dp2*(pw2(py)+pw2(pz));
	  if(p2>0) // The zeromode is set separately below
	    set_mode(p2,mass_sq[fld], &(fnyquist[2*j]), &(fdnyquist[2*j]), 1);
	}
	// Next, copy the values from the nyquist array to the f and fd arrays in the correct processors
	for(j=0; j<n; j++) // Copying first 2*n of fnyquist/fdnyquist into my_rank=0
	{
	  f[fld][j+1][2*k] = fnyquist[2*j];
	  f[fld][j+1][2*k+1] = fnyquist[2*j+1];
	  fd[fld][j+1][2*k] = fdnyquist[2*j];
	  fd[fld][j+1][2*k+1] = fdnyquist[2*j+1];
	}
	j=n; // Start sending Nyquist modes to other processors just after the ones that were stored in the root processor
	for(proc=1; proc<numprocs; proc++) // Sending Nyquist modes to other processors
	{
	  MPI_Send((void*)&fnyquist[2*j], 2*all_n[proc], MPI_FLOAT, proc, proc, MPI_COMM_WORLD);
	  MPI_Send((void*)&fdnyquist[2*j], 2*all_n[proc], MPI_FLOAT, proc, proc+numprocs, MPI_COMM_WORLD);
	  j+=all_n[proc]; // Set j to the starting position for the next processor's data
	}
      } // End tasks for processor 0      
      else // For all other processors, receive the data sent by the root processor
      {
	MPI_Recv((void *)&(f[fld][1][2*k]), 1, skipped_array, 0, my_rank, MPI_COMM_WORLD, &status);
	MPI_Recv((void *)&(fd[fld][1][2*k]), 1, skipped_array, 0, my_rank+numprocs, MPI_COMM_WORLD, &status);
      }
    } // End loop over edges of k.    
    if(my_rank==0) // Set the zeromode on the root processor
    {
      f[fld][1][0] = 0.;
      f[fld][1][1] = 0.;
      fd[fld][1][0] = 0.;
      fd[fld][1][1] = 0.;
    }
    // As of now every processor should have all initialized data in correct order and ready for transform

    // *DEBUG* The option to not use the FFTs is for debugging purposes only. For actual simulations seed should always be positive
    if(seed>=0)
    {    
    
      fftwf_execute(plans_f[fld]);  // PH
      fftwf_execute(plans_fd[fld]); // PH

      // FFTW doesn't divide by the array size in an inverse transform, so that has to be done by hand here.
      for(j=1; j<=n; j++)
	for(k=0; k<N; k++)
	{
	  f[fld][j][k] /= (float)gridsize;
	  fd[fld][j][k] /= (float)gridsize; 
	}
    }
#elif NDIMS==3
    for(i=1; i<=n; i++)
    {
      px = i-1 + local_i_start; // x index in the entire array (not just this processor's array)
      px=(px<=N/2 ? px : px-N); // The momentum in wrap around order. 
      for(j=0; j<N; j++)
      {
	py = (j<=N/2 ? j : j-N); // The momentum in wrap around order. 
	for(k=1; k<N/2; k++) // Modes in the bulk (k>0, k<N/2) are not real and have no complex conjugate mode in the lattice
	{
	  pz = k;
	  p2 = dp2*(pw2(px)+pw2(py)+pw2(pz)); 
	  set_mode(p2, mass_sq[fld], &f[fld][i][j][2*k], &fd[fld][i][j][2*k],0); // Initialize mode
	} //End of loop over k
      } // End of loop over j
    } // End of loop over i

    // Modes with k=0 or k=N/2 are calculated in the nyquist array on the root processor and then copied to the field arrays of the appropriate processors.
    for(k=0;k<=N/2;k+=N/2)
    {
      pz=k;    
      if(my_rank==0) // Initialize all conjugated/real values on the root processor.
      {
	// First set all of the correct values in the nyquist array
	for(i=0;i<N;i++)
	{
	  px=(i<=N/2 ? i : i-N); // The momentum in wrap around order. 
	  iconj=(i==0 ? i : N-i); // The location of the mode conjugate to the mode at location px
	  for(j=N/2+1; j<N; j++) // The "lower half" (j<N/2) will be initialized as the conjugate of the "upper half" (j>N/2)
	  {
	    py = j-N; // The momentum in wrap around order.
	    jconj=-py; // The location of the mode conjugate to the mode at location py
	    p2 = dp2*(pw2(px)+pw2(py)+pw2(pz));
	    set_mode(p2, mass_sq[fld], &fnyquist[i][2*j], &fdnyquist[i][2*j], 0); // Set the mode in the "upper half"
	    // Next set the corresponding "lower half" mode to the complex conjugate of the "upper half" mode
	    fnyquist[iconj][2*jconj] = fnyquist[i][2*j];
	    fnyquist[iconj][2*jconj+1] = -fnyquist[i][2*j+1];
	    fdnyquist[iconj][2*jconj] = fdnyquist[i][2*j];
	    fdnyquist[iconj][2*jconj+1] = -fdnyquist[i][2*j+1];
	  }
	  for(j=0;j<=N/2;j+=N/2) // These two strips have to be handled separately
	  {
	    py=j;
	    jconj=j;
	    if(i==0 || i==N/2) // The points with px, py, and pz each equal to 0 or N/2 are real
	    {
	      p2 = dp2*(pw2(px)+pw2(py)+pw2(pz));
	      if (p2>0.) // The zeromode is set separately below
		set_mode(p2,mass_sq[fld],&fnyquist[i][2*j],&fdnyquist[i][2*j],1);
	    }
	    else if(i>N/2) // All other points on these strips are handled like the conjugate modes above. The modes in the "upper half" (i>N/2) are initialized and then the conjugate modes are set
	    {
	      p2 = dp2*(pw2(px)+pw2(py)+pw2(pz));
	      set_mode(p2, mass_sq[fld], &fnyquist[i][2*j], &fdnyquist[i][2*j], 0); // Set the mode in the "upper half"
	      // Next set the corresponding "lower half" mode to the complex conjugate of the "upper half" mode
	      fnyquist[iconj][2*jconj] = fnyquist[i][2*j];
	      fnyquist[iconj][2*jconj+1] = -fnyquist[i][2*j+1];
	      fdnyquist[iconj][2*jconj] = fdnyquist[i][2*j];
	      fdnyquist[iconj][2*jconj+1] = -fdnyquist[i][2*j+1];
	    }
	  } // End loop where j equals 0 or N/2
	} // End loop over i
	// Next, copy the values from the nyquist array to the work and fd arrays in the correct processors
	for(i=0; i<n; i++) // Copying first n x values of fnyquist/fdnyquist into my_rank=0
   	  for(j=0; j<N; j++)
	  {
	    f[fld][i+1][j][2*k]=fnyquist[i][2*j];
	    f[fld][i+1][j][2*k+1]=fnyquist[i][2*j+1];
	    fd[fld][i+1][j][2*k]=fdnyquist[i][2*j];
	    fd[fld][i+1][j][2*k+1]=fdnyquist[i][2*j+1];
	  }
	i=n; // Start sending Nyquist modes to other processors just after the ones that were stored in the root processor
	for(proc=1; proc<numprocs; proc++) // Sending every n x values into its respective processor
	{
	  MPI_Send((void *) &(fnyquist[i][0]), 2*all_n[proc]*N, MPI_FLOAT, proc, proc, MPI_COMM_WORLD);
	  MPI_Send((void *) &(fdnyquist[i][0]), 2*all_n[proc]*N, MPI_FLOAT, proc, proc+numprocs, MPI_COMM_WORLD);
	  i+=all_n[proc]; // Set i to the starting position for the next processor's data
	}	  
      } // End taks for my_rank=0      
      else // For all other processors, receive the data sent by the root processor
      {
	MPI_Recv((void *)&(f[fld][1][0][2*k]), 1, skipped_array, 0, my_rank, MPI_COMM_WORLD, &status);
	MPI_Recv((void *)&(fd[fld][1][0][2*k]), 1, skipped_array, 0, my_rank+numprocs, MPI_COMM_WORLD, &status);
      }
    } // End loop over edges of k.  
    if(my_rank==0) // Set the zeromode on the root processor
    {
      f[fld][1][0][0] = 0;
      f[fld][1][0][1] = 0;
      fd[fld][1][0][0] = 0;
      fd[fld][1][0][1] = 0;
    }
    //As of now every processor should have all initialized data in correct order and ready for transform
    
    // *DEBUG* The option to not use the FFTs is for debugging purposes only. For actual simulations seed should always be positive
    if(seed>=0)
    {

      fftwf_execute(plans_f[fld]);  // PH
      fftwf_execute(plans_fd[fld]); // PH

      // FFTW doesn't divide by the array size in an inverse transform, so that has to be done by hand here.
      for(i=1; i<=n; i++)
	for(j=0; j<N; j++)
	  for(k=0; k<N; k++)
	  {
	    f[fld][i][j][k] /= (float)gridsize;
	    fd[fld][i][j][k] /= (float)gridsize;
	  }
    }
#endif
    LOOP // Add zeromode
    {
      FIELD(fld) += initial_field_values[fld];
      FIELDD(fld) += initial_field_derivs[fld];
    }
  }// End loop over fields

  // Clean up dynamically allocated arrays and FFTW plans
#if NDIMS>1
  // rfftwnd_mpi_destroy_plan(plan);
  for(fld=0;fld<nflds;fld++) // PH --- Have to destroy all the plans now.
  {
    fftwf_destroy_plan(plans_f[fld]);
    fftwf_destroy_plan(plans_fd[fld]);
  }
  if(my_rank==0)
  {
    free(fnyquist);
    free(fdnyquist);
  }
#endif

  buffers_up_to_date=0; // Buffers have now been set out of synch and will need to be updated before any gradients are calculated  
  modelinitialize(2); // This allows specific models to perform any needed initialization
  save(0); // Save field values and derived quantities
  if(my_rank==0)
  {
    output_parameters(); // Save information about the model and parameters and start the clock for timing the run
    printf("Finished initial conditions\n");
  }
  
  return;
}
  






