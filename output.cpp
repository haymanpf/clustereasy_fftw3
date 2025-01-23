/*
This file contains the functions for calculating and outputting results from the lattice program (e.g. field averages and variances).

The output is stored in files named <name>_ext. By default ext is set to <run_number>.dat where run_number indicates how many times this run has been resumed from previous data. For example the file "means_0.dat" would hold the means of an original run. If that run were then continued to a later time the means of the new run would be stored in "means_1.dat". The default value of ext can also be reset in parameters.h
File buffers are flushed at regular intervals controlled by the global variable checkpoint_interval.

The functions here which should be called externally are output_parameters(float time) and save(int force).
The function output_parameters() creates a file and writes to it the parameters used in the program, as well as storing the clock time that elapses during the run.
The function save() calls all the appropriate functions for calculating and saving derived quantities. Which quantities are saved is controlled by global constants contained in the file parameters.h. Certain quantities such as power spectra and histograms are calculated less often, with a frequency again controlled by the variable checkpoint_interval. If save() is called with force!=0 then all quantities are calculated and file buffers are flushed.
*/

#include "latticeeasy.h"

char name_[500]; // Filenames - set differently by each function to open output files

// Outputs the means of the first nfldsout fields to "means.dat" and the variances (<field^2>-<field>^2) to "variances.dat" as a function of time
// Note that field values are output in program - not physical - units. See documentation for the rescalings used.
void meansvars(int flush)
{
  static FILE *means_,*vars_;
  DECLARE_INDICES
  int fld;
  double av,av_sq,var,temp;

  static int first=1;
  if(first) // Open output files
  {
    if(my_rank==0)
    {
      sprintf(name_,"means%s",ext_);
      means_=fopen(name_,mode_);
      sprintf(name_,"variance%s",ext_);
      vars_=fopen(name_,mode_);
    }
    first=0;
  }

  if(my_rank==0)
  {
    fprintf(means_,"%f",t);
    fprintf(vars_,"%f",t);
  }
  for(fld=0;fld<nfldsout;fld++)
  {
    av=0.;
    var=0.;
    // Calculate field mean
    LOOP
      av += FIELD(fld);
    MPI_Allreduce(&av, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Sum temp from all processors
    av = temp/(double)gridsize; // Convert sum to average.
    av_sq = av*av; // Calculate mean squared for finding variance
    // Calculate variance
    LOOP
      var += pw2(FIELD(fld)) - av_sq;
    MPI_Reduce(&var, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Sum temp from all processors
    var = temp/(double)gridsize; // Convert sum to variance
    if(my_rank==0)
    {
      // Output means and variances to files
      fprintf(means_," %e",av*rescaling);
      fprintf(vars_," %e",var*pw2(rescaling));
      // Check for instability. See if the field has grown exponentially and become non-numerical at any point.
      if(av+FLT_MAX==av || (av!=0. && av/av!=1.)) // These two separate checks between them work on all the compilers I've tested
      {
	printf("Unstable solution developed. Field %d not numerical at t=%f\n",fld,t);
	output_parameters();
	fflush(means_);
	fflush(vars_);
	exit(1);
      }
    } // End of if(my_rank==0)
  } // End of loop over fields
  if(my_rank==0)
  {
    fprintf(means_,"\n");
    fprintf(vars_,"\n");
    if(flush)
    {
      fflush(means_);
      fflush(vars_);
    }
  }
}

// Outputs to "sf.dat" the time and the physical quantities a, adot/a (i.e. Hubble), and adotdot
void scale(int flush)
{
  static FILE *sf_;

  if(my_rank>0) // Only the root processor needs to generate this output
    return;

  static int first=1;
  if(first) // Open output files
  {
    sprintf(name_,"sf%s",ext_);
    sf_=fopen(name_,mode_);
    first=0;
  }

  // Output a, H, and adotdot in physical units using model-dependent rescalings
  fprintf(sf_,"%f %f %e %e\n",t,a,ad*rescale_B*pow(a,rescale_s-1.),pw2(rescale_B)*pow(a,2.*rescale_s)*(ad2+rescale_s*pw2(ad)/a));
  if(flush)
    fflush(sf_);
}

// Calculates and outputs spectra for all output fields
// The files "spectra<field>" record for different momentum bins: momentum, number of grid points in the bin, omega_k, |f_k|^2, |f_k'|^2, n_k, and rho_k
// A separate file, "spectratimes.dat", records the times at which spectra are recorded
void spectra()
{
  static FILE *spectra_[nflds],*spectratimes_; // Output files for power spectra and times at which spectra were taken
  int fld;
  const int maxnumbins=(int)(1.73205*(N/2))+1; // Number of bins (bin spacing=lattice spacing in Fourier space) = sqrt(NDIMS)*(N/2)+1. Set for 3D (i.e. biggest possible).
  int numpoints[maxnumbins]; // Number of points in each momentum bin
  float p[maxnumbins],f2[maxnumbins],fd2[maxnumbins],ndensity[maxnumbins],rhodensity[maxnumbins]; // Values for each bin: Momentum, |f_k|^2, |f_k'|^2, n_k, and rho_k
  int numbins=(int)(sqrt((float)NDIMS)*(N/2))+1; // Actual number of bins for the number of dimensions
  float pmagnitude; // Total momentum (p) in units of lattice spacing, pmagnitude = Sqrt(px^2+py^2+pz^2). This also gives the bin index since bin spacing is set to equal lattice spacing.
  float dp=2.*pi/L; // Size of grid spacing in momentum space
  float fp2,fpd2; // Square magnitude of field (fp2) and derivative (fpd2) for a given mode
  float omega,omegasq; // Omega_p^2 = m^2 + p^2. Used for number and energy density
  float mass_sq[nflds]; // Effective square masses for fields. Used for calculating omega
  // Extra terms used to adjust for variable rescalings in the definitions of omega_p, n_p, and rho_p
  float extra_omega=pow(a,2.*rescale_s) * (a*ad2 + (1.+rescale_s)*pw2(ad)); // Extra term in omega
  float extra_fpd_1=(1.-rescale_r)*ad/a, extra_fpd_2=pow(a,2.+2.*rescale_s); // Terms used to modify f'_p
  float extra_np=pow(a,2.-2.*rescale_r)*pow(dx,6)/(pw2(rescale_A*rescale_B)*2.*L*L*L); // Term used to modify n_p
  float extra_rhop=rescale_B*extra_np;
  int i,j,k,px,py,pz; // px, py, and pz are components of momentum in units of grid spacing
#if NDIMS==1
  float norm1=2.*pi*pw2(L)/pw2(pw2(dx)); // Normalization designed to match spectra to 3D results
  float norm2=pw2(pw2((float)N))/2./pi; // A different normalization is used for number and energy spectra
#elif NDIMS==2
  float norm1=pi*L/pw2(dx); // Normalization designed to match spectra to 3D results
  float norm2=pw2((float)N)/2.; // A different normalization is used for number and energy spectra
#elif NDIMS==3
  float norm1=1.,norm2=1.;
#endif
  int itemp; // Temporary storage used in communication between processors
  float temp; // Temporary storage used in communication between processors
#if NDIMS>1
  // These plans are used by FFTW for doing Fourier Transforms
  static fftwf_plan forward_plans_f[nflds];  // PH --- Need a separate plan for each field and its derivative, and for forward/inverse
  static fftwf_plan forward_plans_fd[nflds]; // PH --- Also note we only need arrays of size nfldsout, but it's not a const int (yet...) so I'll use nflds for now.
  static fftwf_plan inverse_plans_f[nflds];  // PH
  static fftwf_plan inverse_plans_fd[nflds]; // PH
#endif

  static int first=1;
  if(first)
  {
    if(my_rank==0) // Open output files
    {
      for(fld=0;fld<nfldsout;fld++)
      {
	sprintf(name_,"spectra%d%s",fld,ext_);
	spectra_[fld]=fopen(name_,mode_);
      }
      sprintf(name_,"spectratimes%s",ext_);
      spectratimes_=fopen(name_,mode_);
    }
#if NDIMS==2
  for(fld=0;fld<nfldsout;fld++)
  {
    forward_plans_f[fld]  = fftwf_mpi_plan_dft_r2c_2d(N, N, &(f[fld][1][0]), (fftwf_complex *)&(f[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH
    forward_plans_fd[fld] = fftwf_mpi_plan_dft_r2c_2d(N, N, &(fd[fld][1][0]), (fftwf_complex *)&(fd[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE); // PH
    inverse_plans_f[fld]  = fftwf_mpi_plan_dft_c2r_2d(N, N, (fftwf_complex *)&(f[fld][1][0]), &(f[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH
    inverse_plans_fd[fld] = fftwf_mpi_plan_dft_c2r_2d(N, N, (fftwf_complex *)&(fd[fld][1][0]), &(fd[fld][1][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH

  }
#elif NDIMS==3
  for(fld=0;fld<nfldsout;fld++)
  {
    forward_plans_f[fld]  = fftwf_mpi_plan_dft_r2c_3d(N, N, N, &(f[fld][1][0][0]), (fftwf_complex *)&(f[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH
    forward_plans_fd[fld] = fftwf_mpi_plan_dft_r2c_3d(N, N, N, &(fd[fld][1][0][0]), (fftwf_complex *)&(fd[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE); // PH
    inverse_plans_f[fld]  = fftwf_mpi_plan_dft_c2r_3d(N, N, N, (fftwf_complex *)&(f[fld][1][0][0]), &(f[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH
    inverse_plans_fd[fld] = fftwf_mpi_plan_dft_c2r_3d(N, N, N, (fftwf_complex *)&(fd[fld][1][0][0]), &(fd[fld][1][0][0]), MPI_COMM_WORLD, FFTW_ESTIMATE);   // PH
  }
#endif

    first=0;
  }

  effective_mass(mass_sq,NULL); // Calculate effective masses of fields.
  // Include corrections involving scale factor derivatives (see documentation)
  for(fld=0;fld<nfldsout;fld++)
    mass_sq[fld] -= extra_omega;

  // Calculate magnitude of momentum in each bin
  for(i=0;i<numbins;i++)
    p[i]=dp*i;

  for(fld=0;fld<nfldsout;fld++) // Find power spectra for each field
  {
    for(i=0;i<numbins;i++) // Initialize all bins to 0
    {
      numpoints[i]=0; // Number of points in the bin
      f2[i]=0.; // |f_p|^2
      fd2[i]=0.; // |f_p'|^2
      ndensity[i]=0.; // omega |f_p|^2 + 1/omega |f_p'|^2
      rhodensity[i]=0.; // omega^2 |f_p|^2 + |f_p'|^2
    }

#if NDIMS==1
    // Take Fourier Transforms
    fftr1(&(f[fld][1]),N,1,my_rank,numprocs);
    fftr1(&(fd[fld][1]),N,1,my_rank,numprocs);

    // All modes except k=0 and k=N/2 are counted twice
    for(k=0;k<n/2;k++) // Initial and final values of the (complex) array index in momentum space
    {
      pz=my_rank*n/2+k; // z-component of momentum of modes at z=k
      pmagnitude=pz; // Magnitude of momentum of mode in units of momentum grid spacing
      if(pmagnitude==0) continue; // The zeromode and Nyquist mode are handled below
      fp2=pw2(f[fld][2*k+1])+pw2(f[fld][2*k+2]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
      fpd2=pw2(fd[fld][2*k+1])+pw2(fd[fld][2*k+2]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
      numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
      f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
      fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
      fpd2=extra_fpd_2*(pw2(fd[fld][2*k+1]+extra_fpd_1*f[fld][2*k+1])+pw2(fd[fld][2*k+2]+extra_fpd_1*f[fld][2*k+2])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
      if(omegasq > 0.) // Guard against floating point errors
      {
	omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega)/pw2(pmagnitude); // Add this mode's contribution to number density
      }
      rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2)/pw2(pmagnitude); // Add this mode's contribution to energy density
    }

    // The modes k=0 and k=N/2 are on the root processor
    if(my_rank==0)
    {
      for(k=0;k<=N/2;k+=N/2) // "Loop" over the two values k=0 and k=N/2
      {
	pz=k;
	pmagnitude=pz;
	if(k==0) // Amplitudes of k=0 modes
	{
	  fp2=pw2(f[fld][1]);
	  fpd2=pw2(fd[fld][1]);
	}
	else // Modes with k=N/2 are stored in the imaginary component of the k=0 array entry
	{
	  fp2=pw2(f[fld][2]);
	  fpd2=pw2(fd[fld][2]);
	}
	numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
	f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
	fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
	if(k==0)
	  fpd2=extra_fpd_2*(pw2(fd[fld][1]+extra_fpd_1*f[fld][1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	else
	  fpd2=extra_fpd_2*(pw2(fd[fld][2]+extra_fpd_1*f[fld][2])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
	if(pmagnitude==0.) // Avoid floating point errors
	  continue;
	if(omegasq > 0.) // Guard against floating point errors
	{
	  omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	  ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega)/pw2(pmagnitude); // Add this mode's contribution to number density
	}
	rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2)/pw2(pmagnitude); // Add this mode's contribution to energy density
      } // End of loop over k=0 and k=N/2
    } // End of if(my_rank==0)
    // Transform field values back to position space
    fftr1(&(f[fld][1]),N,-1,my_rank,numprocs);
    fftr1(&(fd[fld][1]),N,-1,my_rank,numprocs);
#elif NDIMS==2
    // Take Fourier Transforms
    fftwf_execute(forward_plans_f[fld]);  // PH
    fftwf_execute(forward_plans_fd[fld]); // PH

    // Loop runs over all gridpoints.
    // px and py go over all mode values in wrap-around order, rising from 0 to N/2 and then from -N/2+1 to -1
    for(j=1;j<=n;j++)
    {
      py = j-1 + my_start_position; // First find the index value of i in the entire array
      py = (py<=N/2 ? py : py-N); // Set the x-momentum in wrap-around order

	// Modes with 0<k<N/2 are counted twice to account for the modes f(-p)=f(p)* that aren't explicitly included in the lattice
      for(k=1;k<N/2;k++) 
      {
	pz=k; // z-component of momentum of modes at z=k
	pmagnitude=sqrt(pw2(py)+pw2(pz)); // Magnitude of momentum of mode in units of momentum grid spacing
	fp2=pw2(f[fld][j][2*k])+pw2(f[fld][j][2*k+1]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
	fpd2=pw2(fd[fld][j][2*k])+pw2(fd[fld][j][2*k+1]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
	numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
	f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
	fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
	fpd2=extra_fpd_2*(pw2(fd[fld][j][2*k]+extra_fpd_1*f[fld][j][2*k])+pw2(fd[fld][j][2*k+1]+extra_fpd_1*f[fld][j][2*k+1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
	if(omegasq > 0.) // Guard against floating point errors
	{
	  omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	  ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega)/pmagnitude; // Add this mode's contribution to number density
	}
	rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2)/pmagnitude; // Add this mode's contribution to energy density
      }
      
      // Modes with k=0 are only counted once
      pz=0;
      pmagnitude=sqrt(pw2(py)+pw2(pz));
      fp2=pw2(f[fld][j][0])+pw2(f[fld][j][1]);
      fpd2=pw2(fd[fld][j][0])+pw2(fd[fld][j][1]);
      numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
      f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
      fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
      fpd2=extra_fpd_2*(pw2(fd[fld][j][0]+extra_fpd_1*f[fld][j][0])+pw2(fd[fld][j][1]+extra_fpd_1*f[fld][j][1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
      if(pmagnitude==0.) // Avoid floating point errors
        continue;
      if(omegasq > 0.) // Guard against floating point errors
      {
	omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega)/pmagnitude; // Add this mode's contribution to number density
      }
      rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2)/pmagnitude; // Add this mode's contribution to energy density


      // Modes with k=N/2 are also only counted once
      pz=N/2;
      pmagnitude=sqrt(pw2(py)+pw2(pz));
      fp2=pw2(f[fld][j][N])+pw2(f[fld][j][N+1]);
      fpd2=pw2(fd[fld][j][N])+pw2(fd[fld][j][N+1]);
      numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
      f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
      fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
      fpd2=extra_fpd_2*(pw2(fd[fld][j][N]+extra_fpd_1*f[fld][j][N])+pw2(fd[fld][j][N]+extra_fpd_1*f[fld][j][N+1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
      omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
      if(omegasq > 0.) // Guard against floating point errors
      {
	omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega)/pmagnitude; // Add this mode's contribution to number density                 
      }
      rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2)/pmagnitude; // Add this mode's contribution to energy density   
    } // End of loop over j
    // Transform field values back to position space
    fftwf_execute(inverse_plans_f[fld]);  // PH
    fftwf_execute(inverse_plans_fd[fld]); // PH
        
#elif NDIMS==3
    // Take Fourier Transforms
    fftwf_execute(forward_plans_f[fld]);  // PH
    fftwf_execute(forward_plans_fd[fld]); // PH

    // Loop runs over all gridpoints.
    // px and py go over all mode values in wrap-around order, rising from 0 to N/2 and then from -N/2+1 to -1
    for(i=1;i<=n;i++)
    {
      px = i-1 + my_start_position; // First find the index value of i in the entire array
      px = (px<=N/2 ? px : px-N); // Set the x-momentum in wrap-around order
      for(j=0;j<N;j++)
      {
	py=(j<=N/2 ? j : j-N); // y-component of momentum of modes at y=j

	// Modes with 0<k<N/2 are counted twice to account for the modes f(-p)=f(p)* that aren't explicitly included in the lattice
	for(k=1;k<N/2;k++) 
	{
	  pz=k; // z-component of momentum of modes at z=k
	  pmagnitude=sqrt(pw2(px)+pw2(py)+pw2(pz)); // Magnitude of momentum of mode in units of momentum grid spacing
	  fp2=pw2(f[fld][i][j][2*k])+pw2(f[fld][i][j][2*k+1]); // |Re(f_k)|^2 + |Im(f_k)|^2 for mode
	  fpd2=pw2(fd[fld][i][j][2*k])+pw2(fd[fld][i][j][2*k+1]); // |Re(f_k')|^2 + |Im(f_k')|^2 for mode
	  numpoints[(int)pmagnitude] += 2; // Iterate the count of points in this bin
	  f2[(int)pmagnitude] += 2.*fp2; // Add the power of this mode to the bin
	  fd2[(int)pmagnitude] += 2.*fpd2; // Add the power of the mode derivative to the bin
	  fpd2=extra_fpd_2*(pw2(fd[fld][i][j][2*k]+extra_fpd_1*f[fld][i][j][2*k])+pw2(fd[fld][i][j][2*k+1]+extra_fpd_1*f[fld][i][j][2*k+1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	  omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
	  if(omegasq > 0.) // Guard against floating point errors
	  {
	    omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	    ndensity[(int)pmagnitude] += 2.*extra_np*(fp2*omega + fpd2/omega); // Add this mode's contribution to number density
	  }
	  rhodensity[(int)pmagnitude] += 2.*extra_rhop*(fp2*omegasq + fpd2); // Add this mode's contribution to energy density
	}
	
	// Modes with k=0 are only counted once
	pz=0;
	pmagnitude=sqrt(pw2(px)+pw2(py)+pw2(pz));
	fp2=pw2(f[fld][i][j][0])+pw2(f[fld][i][j][1]);
	fpd2=pw2(fd[fld][i][j][0])+pw2(fd[fld][i][j][1]);
	numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin
	f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin
	fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin
	fpd2=extra_fpd_2*(pw2(fd[fld][i][j][0]+extra_fpd_1*f[fld][i][j][0])+pw2(fd[fld][i][j][1]+extra_fpd_1*f[fld][i][j][1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2
	if(omegasq > 0.) // Guard against floating point errors
	{
	  omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)
	  ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega); // Add this mode's contribution to number density
	}
	rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2); // Add this mode's contribution to energy density


	// Modes with k=N/2 are also only counted once
	pz=N/2;
	pmagnitude=sqrt(pw2(px)+pw2(py)+pw2(pz));
	fp2=pw2(f[fld][i][j][N])+pw2(f[fld][i][j][N+1]);
	fpd2=pw2(fd[fld][i][j][N])+pw2(fd[fld][i][j][N+1]);
	numpoints[(int)pmagnitude]++; // Iterate the count of points in this bin                                                            
	f2[(int)pmagnitude] += fp2; // Add the power of this mode to the bin                                                                
	fd2[(int)pmagnitude] += fpd2; // Add the power of the mode derivative to the bin                                                    
	fpd2=extra_fpd_2*(pw2(fd[fld][i][j][N]+extra_fpd_1*f[fld][i][j][N])+pw2(fd[fld][i][j][N]+extra_fpd_1*f[fld][i][j][N+1])); // Adjusted value of |f'_p|^2 used in n_p and rho_p
	omegasq = fabs(pw2(dp*pmagnitude) + mass_sq[fld]); // Calculate omega_p^2                                                           
        if(omegasq > 0.) // Guard against floating point errors                                                                             
        {
	  omega = sqrt(omegasq); // Omega = Sqrt(p^2 + m^2)                                                                                 
          ndensity[(int)pmagnitude] += extra_np*(fp2*omega + fpd2/omega); // Add this mode's contribution to number density                 
	}
	rhodensity[(int)pmagnitude] += extra_rhop*(fp2*omegasq + fpd2); // Add this mode's contribution to energy density   

      } // End of loop over j
    } // End of loop over i
    // Transform field values back to position space
    fftwf_execute(inverse_plans_f[fld]);  // PH
    fftwf_execute(inverse_plans_fd[fld]); // PH
#endif
#if NDIMS>1
    LOOP // rescale by total number of points since FFTW does not do this.
    {
      FIELD(fld) /= gridsize;
      FIELDD(fld) /= gridsize;
    }
#endif

    // The values of power, number, and energy have all been tabulated. Now they need to be gathered and output. 
    for(i=0;i<numbins;i++)
    {
      // Gather all data to the root processor
      MPI_Reduce(&(numpoints[i]),&itemp,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
      if(my_rank==0) numpoints[i]=itemp;
      MPI_Reduce(&(f2[i]),&temp,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      if(my_rank==0 && numpoints[i]>0) f2[i]=temp/numpoints[i];
      MPI_Reduce(&(fd2[i]),&temp,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      if(my_rank==0 && numpoints[i]>0) fd2[i]=temp/numpoints[i];
      MPI_Reduce(&(ndensity[i]),&temp,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      if(my_rank==0 && numpoints[i]>0) ndensity[i]=temp/numpoints[i];
      MPI_Reduce(&(rhodensity[i]),&temp,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      if(my_rank==0 && numpoints[i]>0) rhodensity[i]=temp/numpoints[i];
      // Print results
      if(my_rank==0)
      {
	omegasq = fabs(pw2(p[i]) + mass_sq[fld]); // Omega and p for each bin are recorded at the bottom of the bin (i.e. the lowest momentum)
	// Output the momentum, number of points, omega, and calculated spectra for each bin
	fprintf(spectra_[fld],"%e %d %e %e %e %e %e\n",
		p[i],numpoints[i],omegasq,norm1*f2[i],norm1*fd2[i],norm2*ndensity[i],norm2*rhodensity[i]);
      }
    } // End loop over bins
    if(my_rank==0)
    {
      fprintf(spectra_[fld],"\n");
      fflush(spectra_[fld]);
    }
  } // End of loop over fields
  
  if(my_rank==0)
  {
    fprintf(spectratimes_,"%f\n",t); // Output time at which power spectra were recorded
    fflush(spectratimes_);
  }

  buffers_up_to_date=0; // Buffers have now been set out of synch and will need to be updated before any gradients are calculated  

  return;
}

// Outputs the components of the total energy as a function of time
// The output file "energy" contains t, kinetic (time derivative) energy for each field, gradient energy for each field, and then all potential terms
// The file "conservation" contains rho/rho_initial for no expansion and the ratio of H^2 to 8 pi/3 rho for an expanding universe
// Note that energy output is not in physical units. See the documentation for conversions.
void energy()
{
  static FILE *energy_,*conservation_;
  static float totalinitial; // Initial value of energy density (used for checking conservation in Minkowski space)
  DECLARE_INDICES
  int fld;
  float var,vard,ffd; // Averaged field values over the grid (defined below)
  float deriv_energy,grad_energy,pot_energy,total=0.,temp; // Total gives the total value of rho(t) for checking energy conservation

  static int first=1;
  if(first && my_rank==0) // Open output files (first is set to zero at the bottom of the function)
  {
    sprintf(name_,"energy%s",ext_);
    energy_=fopen(name_,mode_);
    if(expansion!=1) // For power-law expansion don't calculate conservation
    {
      sprintf(name_,"conservation%s",ext_);
      conservation_=fopen(name_,mode_);
    }
  } // The variable first is used again at the end of the function, where it is then set to 0

  if(my_rank==0)
    fprintf(energy_,"%f",t); // Output time

  // Calculate and output kinetic (time derivative) energy
  for(fld=0;fld<nflds;fld++) // Note that energy is output for all fields regardless of the value of nfldsout
  {
    var=0.;
    vard=0.;
    ffd=0.;
    LOOP
    {
      var += pw2(FIELD(fld)); // Sum over squared field values
      vard += pw2(FIELDD(fld)); // Sum over squared field derivatives
      ffd += FIELD(fld)*FIELDD(fld); // Sum over product of field and derivative
    }
    // Convert sums to averages
    var = var/(float)gridsize;
    vard = vard/(float)gridsize;
    ffd = ffd/(float)gridsize;
    deriv_energy = .5*vard - rescale_r*(ad/a)*ffd + .5*pw2(rescale_r)*pw2(ad/a)*var; // Extra terms account for model-dependent rescalings
    MPI_Reduce(&deriv_energy, &temp, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // Average deriv_energy from all processors
    total += temp;
    if(my_rank==0)
      fprintf(energy_," %e",temp);
  }

  // Calculate and output gradient energy
  for(fld=0;fld<nflds;fld++)
  {
    grad_energy = pow(a,-2.*(rescale_s+1.))*gradient_energy(fld);
    total += grad_energy;
    if(my_rank==0)
      fprintf(energy_," %e",grad_energy);
  }

  // Calculate and output potential energy
  for(i=0;i<num_potential_terms;i++)
  {
    pot_energy = potential_energy(i,NULL); // Model dependent function for calculating potential energy terms
    total += pot_energy;
    if(my_rank==0)
      fprintf(energy_," %e",pot_energy);
  }

  if(my_rank==0)
  {
    fprintf(energy_,"\n");
    fflush(energy_);
  }

  // Energy conservation
  if(first) // In Minkowski space record the initial value of the energy to be used for checking energy conservation
  {
    if(expansion==0 && my_rank==0)
      totalinitial=total;
    first=0; // Regardless of the expansion set first to 0 so file streams aren't opened again.
  }
  if(expansion!=1 && my_rank==0) // Conservation isn't checked for power law expansion
  {
    if(expansion==0) // In Minkowski space the file conservation_ records the ratio of rho(t) to rho_initial
      fprintf(conservation_,"%f %f\n",t,total/totalinitial);
    else // In an expanding universe the file conservation_ records the ratio of H^2(t) to 8 pi/3 rho(t)
      fprintf(conservation_,"%f %f\n",t,3.*pw2(rescale_A)/(8.*pi)*pow(a,2.*(rescale_r-1.))*pw2(ad)/total);
    fflush(conservation_);
  }
}

// Outputs histograms for all output fields
// The histograms are stored in files "histogram<filenumber>" where <filenumber> runs from 0 to nfldsout.
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms()
{
  static FILE *histogram_[nflds],*histogramtimes_;
  int i=0,j=0,k=0,fld;
  int binnum; // Index of bin for a given field value
  float binfreq[nbins]; // The frequency of field values occurring within each bin
  float bmin,bmax,df,temp; // Minimum and maximum field values for each field and spacing (in field values) between bins
  int numpts,tempi; // Count the number of points in the histogram for each field. (Should be all lattice points unless explicit field limits are given.)

  static int first=1;
  if(first) // Open output files
  {
    if(my_rank==0)
    {
      for(fld=0;fld<nfldsout;fld++)
      {
	sprintf(name_,"histogram%d%s",fld,ext_);
	histogram_[fld]=fopen(name_,mode_);
      }
      sprintf(name_,"histogramtimes%s",ext_);
      histogramtimes_=fopen(name_,mode_);
    }
    first=0;
  }

  if(my_rank==0)
    fprintf(histogramtimes_,"%f",t); // Output time at which histograms were recorded
  
  for(fld=0;fld<nfldsout;fld++) // Main loop. Generate histogram for each field.
  {
    // Find the minimum and maximum values of the field
    if(histogram_max==histogram_min) // If no explicit limits are given use the current field values
    {
      i=0;j=0;k=0;
      bmin=FIELD(fld);
      bmax=bmin;
      LOOP
      {
        bmin = (FIELD(fld)<bmin ? FIELD(fld) : bmin);
        bmax = (FIELD(fld)>bmax ? FIELD(fld) : bmax);
      }
      MPI_Allreduce(&bmin, &temp, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      bmin=temp;
      MPI_Allreduce(&bmax, &temp, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      bmax=temp;
    }
    else
    {
      bmin=histogram_min;
      bmax=histogram_max;
    }
  
    // Find the difference (in field value) between successive bins
    df=(bmax-bmin)/(float)(nbins); // bmin will be at the bottom of the first bin and bmax at the top of the last
  
    // Initialize all frequencies to zero
    for(i=0;i<nbins;i++)
      binfreq[i]=0.;

    // Iterate over grid to determine bin frequencies
    numpts=0;
    LOOP
    {
      binnum=(int)((FIELD(fld)-bmin)/df); // Find index of bin for each value
      if(FIELD(fld)==bmax) // The maximal field value is at the top of the highest bin
        binnum=nbins-1;
      if(binnum>=0 && binnum<nbins) // Increment frequency in the appropriate bin
      {
        binfreq[binnum]++;
        numpts++;
      }
    } // End of loop over grid

    // Gather results
    for(i=0;i<nbins;i++)
    {
      temp=binfreq[i];
      MPI_Reduce(&temp, &(binfreq[i]), 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    tempi=numpts;
    MPI_Reduce(&tempi, &numpts, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(my_rank==0)
    {
      // Output results
      for(i=0;i<nbins;i++)
	fprintf(histogram_[fld],"%e\n",binfreq[i]/(float)numpts); // Output bin frequency normalized so the total equals 1
      fprintf(histogram_[fld],"\n"); // Stick a blank line between times to make the file more readable
      fflush(histogram_[fld]);
      fprintf(histogramtimes_," %e %e",bmin*rescaling,df*rescaling); // Output the starting point and stepsize for the bins at each time
    }
  } // End loop over fields

  if(my_rank==0)
  {
    fprintf(histogramtimes_,"\n");
    fflush(histogramtimes_);
  }
}

// Calculate two dimensional histograms of pairs of fields
// The histograms are stored in files "histogram2d<filenumber>_<filenumber>".
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms_2d()
{
  static FILE **histogram_,*histogramtimes_; // histogram_ is an array of pointers to output files
  int i,j,k,pair; // pair iterates over pairs of fields for which histograms are being recorded
  int fld1,fld2; // Indices of the fields for a given histogram
  int binnum[2]; // Indices of bins for a given pair of field values
  float binfreq[nbins0][nbins1]; // The frequency of field values occurring within each bin
  float bmin[2],bmax[2],df[2],temp; // Minimum and maximum field values for each field and spacing (in field values) between bins

  static int first=1,numhists; // numhists is the numbers of 2d histograms being calculated
  if(first) // Open output files
  {
    numhists = (sizeof(hist2dflds)/sizeof(hist2dflds[0])/2); // Number of pairs in field list
    if(my_rank==0)
    {
      histogram_ = new FILE *[numhists]; // Allocate an array of FILE pointers
      for(pair=0;pair<numhists;pair++)
      {
	sprintf(name_,"histogram2d%d_%d%s",hist2dflds[2*pair],hist2dflds[2*pair+1],ext_);
	histogram_[pair]=fopen(name_,mode_);
      }
      sprintf(name_,"histogram2dtimes%s",ext_);
      histogramtimes_=fopen(name_,mode_);
    }
    first=0;
  }

  if(my_rank==0)
    fprintf(histogramtimes_,"%f",t); // Output time at which 2d histograms were recorded

  for(pair=0;pair<numhists;pair++) // Main loop. Generate histogram for each field pair
  {
    fld1=hist2dflds[2*pair]; // Find indices of fields for 2d histogram
    fld2=hist2dflds[2*pair+1];
    // Find the minimum and maximum values of the field
    if(histogram2d_min==histogram2d_max) // If no explicit limits are given use the current field values
    {
      i=0;j=0;k=0;
      bmin[0]=FIELD(fld1);
      bmax[0]=bmin[0];
      bmin[1]=FIELD(fld2);
      bmax[1]=bmin[1];
      LOOP
      {
        bmin[0] = (FIELD(fld1)<bmin[0] ? FIELD(fld1) : bmin[0]);
        bmax[0] = (FIELD(fld1)>bmax[0] ? FIELD(fld1) : bmax[0]);
        bmin[1] = (FIELD(fld2)<bmin[1] ? FIELD(fld2) : bmin[1]);
        bmax[1] = (FIELD(fld2)>bmax[1] ? FIELD(fld2) : bmax[1]);
      }
      MPI_Allreduce(&(bmin[0]), &temp, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      bmin[0]=temp;
      MPI_Allreduce(&(bmax[0]), &temp, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      bmax[0]=temp;
      MPI_Allreduce(&(bmin[1]), &temp, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      bmin[1]=temp;
      MPI_Allreduce(&(bmax[1]), &temp, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      bmax[1]=temp;
    }
    else
    {
      bmin[0]=histogram2d_min;
      bmax[0]=histogram2d_max;
      bmin[1]=histogram2d_min;
      bmax[1]=histogram2d_max;
    }

    // Find the difference (in field value) between successive bins
    df[0]=(bmax[0]-bmin[0])/(float)(nbins0);
    df[1]=(bmax[1]-bmin[1])/(float)(nbins1);
    
    // Initialize all frequencies to zero
    for(i=0;i<nbins0;i++)
      for(j=0;j<nbins1;j++)
        binfreq[i][j]=0.;
  
    // Iterate over grid to determine bin frequencies
    LOOP
    {
      binnum[0]=(int)((FIELD(fld1)-bmin[0])/df[0]); // Find index of bin for each value
      binnum[1]=(int)((FIELD(fld2)-bmin[1])/df[1]);
      if(FIELD(fld1)==bmax[0]) binnum[0]=nbins0-1; // The maximal field value is at the top of the highest bin
      if(FIELD(fld2)==bmax[1]) binnum[1]=nbins1-1; // The maximal field value is at the top of the highest bin
      if(binnum[0]>=0 && binnum[0]<nbins0 && binnum[1]>=0 && binnum[1]<nbins1)
	binfreq[binnum[0]][binnum[1]]++;
    } // End of loop over grid

    // Gather results
    for(i=0;i<nbins0;i++)
      for(j=0;j<nbins1;j++)
      {
	temp=binfreq[i][j];
	MPI_Reduce(&temp, &(binfreq[i][j]), 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      }

    // Output results
    if(my_rank==0)
    {
      for(i=0;i<nbins0;i++)
	for(j=0;j<nbins1;j++)
	  fprintf(histogram_[pair],"%e\n",binfreq[i][j]/(float)gridsize);
      fprintf(histogram_[pair],"\n"); // Stick a blank line between times to make the file more readable
      fflush(histogram_[pair]);
      fprintf(histogramtimes_," %e %e %e %e",bmin[0]*rescaling,df[0]*rescaling,bmin[1]*rescaling,df[1]*rescaling); // Output the starting point and stepsize for the bins at each time
    }
  } // End of loop over pairs

  if(my_rank==0)
  {
    fprintf(histogramtimes_,"\n");
    fflush(histogramtimes_);
  }
}


// This function outputs the values of the fields on a slice of the lattice
// Note that if slicedim=NDIMS in the MPI version then the sliceaverage parameter will be ignored.
inline void slices()
{
  static FILE *slices_[nflds],*slicetimes_;
  static int adjusted_slicedim=slicedim; // Number of dimensions to include on slice
  static int final=(slicelength-1)*sliceskip; // This gives the last value that will be included in the slice. (Note that the serial version calculates a point beyond that and then uses i<final, but this way is easier for finding the last processor that will be needed for a given slice.)
  int fld,i=0,j=0,k=0;
  int x=0,y=0,z=0; // A separate set of array indices is needed for averaging
  int numpts; // Used for keeping track of how many points are being averaged in each output
  float value; // Field value to be output
  // MPI parameters
  int proc=my_rank;
  static int max_rank=my_rank; // Highest-rank process needed for output of slices
  static int local_gridsize=0; // Size of the array to be used from each processor, and the maximum of all of these
  static int istart=1;; // Starting position in the array for each processor
  int maximum_i; // Ending position in the array for each processor
  int proc_gridsize; // Size of the array received by the root from each processor
  float *fpoint; // Pointer for referencing different parts of the field array
  static int abort_slices=0; // Used to abort function if the root processor was unable to cast the array it needs to for the data
  MPI_Status status;

  static int first=1;
  if(first) // Open output files and initialize
  {
    // Check if slices needs to be aborted because the root processor couldn't cast the array it needed
    if(my_rank==0 && slicedim>=NDIMS && fstore==NULL)
      abort_slices=1;
    MPI_Bcast(&abort_slices,1,MPI_INT,0,MPI_COMM_WORLD);
    if(abort_slices==1) return;

    // Open files
    if(my_rank==0)
    {
      for(fld=0;fld<nfldsout;fld++)
      {
	sprintf(name_,"slices%d%s",fld,ext_);
	slices_[fld]=fopen(name_,mode_);
      }
      sprintf(name_,"slicetimes%s",ext_);
      slicetimes_=fopen(name_,mode_);
    }

    if(adjusted_slicedim>NDIMS) // Set dimensionality for slices
      adjusted_slicedim=NDIMS;
    if(final>=N) // Set the final location in the grid that's used for the slices
      final=N-1;

    if(adjusted_slicedim==NDIMS) // In this case MPI communication may be needed
    {
      // Find position in this processor's lattice to start output
      istart = sliceskip + 1 - my_start_position % sliceskip;
      if(istart==sliceskip+1) istart=1; // The formula above will be incorrect if you should start at the first row
      // Find position in this processor's lattice to end output
      maximum_i = n + 1 - (my_start_position + n)%sliceskip;
      if(maximum_i == n+1) maximum_i -= sliceskip; // The formula above will be incorrect in this one case
      while(maximum_i+my_start_position-1>final) maximum_i -= sliceskip; // Make sure the final position in the local lattice isn't past the final position overall
      // Find the max_rank used for the slices
      if(maximum_i<1 || my_start_position+istart-1>final) proc=0;
      MPI_Allreduce(&proc, &max_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // The highest ranked processor that didn't set proc=0 above is the max_rank
      // Calculate the size of the data to be sent from each processor
      if(my_rank<=max_rank) local_gridsize = maximum_i-istart+1;
      if(NDIMS>1) local_gridsize *= (N+2); // In 2D or 3D the last dimension has size N+2 (for FFTW padding)
      if(NDIMS>2) local_gridsize *= N; // In 3D the middle dimension has size N
    } // End of if(adjusted_slicedims==NDIMS)
    first=0; // Ensure that initialization tasks (e.g. opening slicetimes file) are only done once.
  } // End of if(first)

  if(abort_slices==1) return;
  
  if(adjusted_slicedim==NDIMS && max_rank>0) // In this case multiple processors are needed. Note that in this case the sliceaverage parameter is ignored
  {
    for(fld=0;fld<nfldsout;fld++)
    {
#if NDIMS==1
      fpoint = &(f[fld][istart]);
#elif NDIMS==2
      fpoint = &(f[fld][istart][0]);
#elif NDIMS==3
      fpoint = &(f[fld][istart][0][0]);
#endif
      if(my_rank>0 && my_rank<=max_rank) // All processors above rank 0 should send their data to the root processor
	MPI_Send((void *)fpoint,local_gridsize,MPI_FLOAT,0,my_rank+fld*max_rank,MPI_COMM_WORLD);
      else if(my_rank==0) // The root processor should receive all the data and print it to the files
      {
	for(proc=0;proc<=max_rank;proc++)
	{
	  if(proc==0)
	    proc_gridsize = local_gridsize;
	  else
	  {
	    MPI_Recv((void *)fstore,fstore_size,MPI_FLOAT,proc,proc+fld*max_rank,MPI_COMM_WORLD,&status);
	    MPI_Get_count(&status,MPI_FLOAT,&proc_gridsize); // Find the size of the array sent by this processor	    
	    fpoint = fstore;
	  }
#if NDIMS==1
	  maximum_i = proc_gridsize;
	  for(i=0;i<maximum_i;i+=sliceskip)
	    fprintf(slices_[fld],"%e\n",fpoint[i]*rescaling);
#elif NDIMS==2
	  maximum_i = proc_gridsize/(N+2);
	  for(i=0;i<maximum_i;i+=sliceskip)
	    for(j=0;j<=final;j+=sliceskip)
	      fprintf(slices_[fld],"%e\n",fpoint[j+i*(N+2)]*rescaling);
#elif NDIMS==3
	  maximum_i = proc_gridsize/N/(N+2);
	  for(i=0;i<maximum_i;i+=sliceskip)
	    for(j=0;j<=final;j+=sliceskip)
	      for(k=0;k<=final;k+=sliceskip)
		fprintf(slices_[fld],"%e\n",fpoint[k+j*(N+2)+i*N*(N+2)]*rescaling);
#endif
	} // End of loop over procs
	fprintf(slices_[fld], "\n");
	fflush(slices_[fld]);
      } // End of if(my_rank==0)
    } // End of loop over fields
  } // End of case where more than one processor is needed
  else if(my_rank==0) // If adjusted_slicedim<NDIMS (or max_rank=0) everything can be done at the root processor
  {
    for(fld=0;fld<nfldsout;fld++)
    {
      if(adjusted_slicedim == 1)
      {
	for(k=0;k<=final;k+=sliceskip)
	{
	  if(sliceaverage==1) // Average over all "skipped" points
	  {
	    value=0.;
	    numpts=0;
	    for(z=k;z<k+sliceskip && z<N;z++)
	    {
#if NDIMS==1
	      if(z>=n) continue; // Make sure you don't go beyond the end of the points stored on this lattice. This situation should only occur if numprocs>1 but final is set low enough that only the first processor is needed, and sliceskip>1 and slice averaging is turned on. In that case this might result in one point in the slice having a different value in the serial and parallel versions because a different number of points will have been averaged.
	      value += f[fld][z+1]; // The +1 adjusts for the offset in the first dimension
#elif NDIMS==2
	      value += f[fld][1][z];
#elif NDIMS==3
	      value += f[fld][1][0][z];
#endif
	      numpts++;
	    }
	    if(numpts>0) value /= (float)numpts;
	    else printf("Division by zero attempted in slices routine at point k=%d\n",k);
	  }
	  else // ...or just output field values at the sampled points
	  {
#if NDIMS==1
	    value = f[fld][k+1]; // The +1 adjusts for the offset in the first dimension
#elif NDIMS==2
	    value = f[fld][1][k];
#elif NDIMS==3
	    value = f[fld][1][0][k];
#endif
	  }
	  fprintf(slices_[fld],"%e\n",value*rescaling);
	}
      } // End of if(adjusted_slicedim==1)
      else if(adjusted_slicedim==2)
      {
	for(j=0;j<=final;j+=sliceskip)
	  for(k=0;k<=final;k+=sliceskip)
	  {
	    if(sliceaverage==1) // Average over all "skipped" points
	    {
	      value=0.;
	      numpts=0;
	      for(y=j;y<j+sliceskip && y<N;y++)
		for(z=k;z<k+sliceskip && z<N;z++)
		{
#if NDIMS==2
		  if(y>=n) continue; // Don't go beyond the local lattice. See comments above
		  value += f[fld][y+1][z]; // The +1 adjusts for the offset in the first dimension
#elif NDIMS==3
		  value += f[fld][1][y][z];
#endif
		  numpts++;
		}
	      if(numpts>0) value /= (float)numpts;
	      else printf("Division by zero attempted in slices routine at point j=%d, k=%d\n",j,k);
	    }
	    else // ...or just output field values at the sampled points
	    {
#if NDIMS==2
	      value = f[fld][j+1][k]; // The +1 adjusts for the offset in the first dimension
#elif NDIMS==3
	      value = f[fld][1][j][k];
#endif
	    }
	    fprintf(slices_[fld],"%e\n",value*rescaling);
	  }
      } // End of if(adjusted_slicedim==2)
      else if(adjusted_slicedim==3)
      {
	for(i=0;i<=final;i+=sliceskip)
	  for(j=0;j<=final;j+=sliceskip)
	    for(k=0;k<=final;k+=sliceskip)
	    {
	      if(sliceaverage==1) // Average over all "skipped" points
	      {
		value=0.;
		numpts=0;
		for(x=i;x<i+sliceskip && x<n;x++)
		  for(y=j;y<j+sliceskip && y<N;y++)
		    for(z=k;z<k+sliceskip && z<N;z++)
		    {
#if NDIMS==3 // This code should only be reached when NDIMS=3, but the compiler would complain without this line
		      value += f[fld][x+1][y][z]; // The +1 adjusts for the offset in the first dimension
#endif
		      numpts++;
		    }
		if(numpts>0) value /= (float)numpts;
		else printf("Division by zero attempted in slices routine at point i=%d, j=%d, k=%d\n",i,j,k);
	      }
	      else // ...or just output field values at the sampled points
#if NDIMS==3 // This code should only be reached when NDIMS=3, but the compiler would complain without this line
		value = f[fld][i+1][j][k]; // The +1 adjusts for the offset in the first dimension
#endif
	      fprintf(slices_[fld],"%e\n",value*rescaling);
	    }
      } // End of if(adjusted_slicedim==3)
      fprintf(slices_[fld],"\n");
      fflush(slices_[fld]);
    } // End of loop over fields
  } // End of case where slices are entirely at root processor
  
  if(my_rank==0)
  {
    fprintf(slicetimes_,"%f\n",t);
    fflush(slicetimes_);
  }
}

// Output an image of all fields and derivatives on the lattice (and a few other variables) to a binary file
void checkpoint()
{
  static FILE *grid_; // File for storing lattice images
  // The following variables are all used for keeping track of when to close one field value file and begin writing to another.
  static int numtimes=sizeof(store_lattice_times)/sizeof(float); // Number of times at which to switch grid image files
  static int current=0; // Index of time value at which to switch files
  static int open=0; // Indicates whether grid image file is open. (It should be unless the program failed to open the file.)
  int itime; // Integer value of the time up to which a file will be used. Used for generating file names.
  char filename[500];
  static int first=1;
  // MPI parameters
  MPI_Status status;
  int i=0,j=0,fld,proc;
  float *fpoint; // Pointer used for sending data to the root processor
  int proc_gridsize, proc_n, local_gridsize=n; // Final i value of the grid received by root, size of grid received by root, and size of grid being sent by other processor
  if(NDIMS>1) local_gridsize *= (N+2); // In 2D or 3D the last dimension has size N+2 (for FFTW padding)
  if(NDIMS>2) local_gridsize *= N; // In 3D the middle dimension has size N

  // Open grid image file
  if(my_rank==0 && fstore!=NULL) // If the fstore array wasn't successfully cast the function will be aborted
  {
    if(first) // The first time the function is called initialize naming scheme
    {
      if(numtimes>0 && store_lattice_times[0]==0.)
	numtimes=0;
      if(numtimes==0) // If no intermediate times are being output simply call the file grid.img
	grid_=fopen("grid.img","wb");
      else // Otherwise label file by the final time it will record data at.
      {
	sprintf(filename,"grid%d.img",(int)store_lattice_times[0]);
	grid_=fopen(filename,"wb");
      }
      first=0;
    }
    else if(current<numtimes && t>store_lattice_times[current]) // If one of the times listed above has been passed switch to another field value file.
    {
      if(open)
	fclose(grid_);
      current++;
      itime = (current<numtimes ? (int)store_lattice_times[current] : (int)tf); // After last time indicated name file with final time of run
      sprintf(filename,"grid%d.img",itime); // Files are labeled by the final time when they will be used, not by the time when they are opened
      grid_=fopen(filename,"wb");
    }

    if(grid_==NULL)
    {
      printf("Error: Grid checkpointing file not open\n");
      open=0;
    }
    else
      open=1;
  }
  MPI_Bcast(&open,1,MPI_INT,0,MPI_COMM_WORLD); // Let all processes know if the grid image successfully opened or not.
  if(!open) // If the grid image file wasn't opened successfully simply exit
    return;

  if(my_rank==0) // Record general information
  {
    rewind(grid_);
    fwrite(&run_number,sizeof(run_number),1,grid_);
    fwrite(&t,sizeof(t),1,grid_);
    fwrite(&a,sizeof(a),1,grid_);
    fwrite(&ad,sizeof(ad),1,grid_);
  }

  for(fld=0;fld<2*nflds;fld++) // The first half is for fields, the second is for field derivatives
  {
#if NDIMS==1
    fpoint = (fld<nflds ? &(f[fld][1]) : &(fd[fld-nflds][1]));
#elif NDIMS==2
    fpoint = (fld<nflds ? &(f[fld][1][0]) : &(fd[fld-nflds][1][0]));
#elif NDIMS==3
    fpoint = (fld<nflds ? &(f[fld][1][0][0]) : &(fd[fld-nflds][1][0][0]));
#endif
    if(my_rank>0) // All processors above rank 0 should send their data to the root processor
      MPI_Send((void *)fpoint,local_gridsize,MPI_FLOAT,0,my_rank+fld*numprocs,MPI_COMM_WORLD);
    else // Root processor should receive data from all others and write it to file
    {
      for(proc=0;proc<numprocs;proc++)
      {
	if(proc==0) // For data at the root processor simply read it from the field array
	  proc_gridsize=local_gridsize;
	else // For other processors receive the data
	{
	  MPI_Recv((void *)fstore,fstore_size,MPI_FLOAT,proc,proc+fld*numprocs,MPI_COMM_WORLD,&status);
	  MPI_Get_count(&status,MPI_FLOAT,&proc_gridsize); // Find the size of the array sent by this processor	    
	  fpoint=fstore;
	}
#if NDIMS==1
	fwrite(fpoint,sizeof(float),proc_gridsize,grid_);
#elif NDIMS==2
	proc_n = proc_gridsize/(N+2);
	for(i=0;i<proc_n;i++) // Output each row (i value) separately to avoid outputting the padding values in the second dimension
	  fwrite(fpoint+i*(N+2),sizeof(float),N,grid_);
#elif NDIMS==3
	proc_n = proc_gridsize/N/(N+2);
	for(i=0;i<proc_n;i++)
	  for(j=0;j<N;j++)
	    fwrite(fpoint+j*(N+2)+i*N*(N+2),sizeof(float),N,grid_);
#endif
      } // End of loop over processors
    } // End of block of code for root processor
  } // End loop over fields
}

// Convert a time in seconds to a more readable form and print the results
void readable_time(int t, FILE *info_)
{
  int tminutes=60,thours=60*tminutes,tdays=24*thours;

  if(t==0)
  {
    fprintf(info_,"less than 1 second\n");
    return;
  }

  // Days
  if(t>tdays)
  {
    fprintf(info_,"%d days",t/tdays);
    t = t%tdays;
    if(t>0)
      fprintf(info_,", ");
  }  
  // Hours
  if(t>thours)
  {
    fprintf(info_,"%d hours",t/thours);
    t = t%thours;
    if(t>0)
      fprintf(info_,", ");
  }
  // Minutes
  if(t>tminutes)
  {
    fprintf(info_,"%d minutes",t/tminutes);
    t = t%tminutes;
    if(t>0)
      fprintf(info_,", ");
  }
  // Seconds
  if(t>0)
    fprintf(info_,"%d seconds",t);
  fprintf(info_,"\n");
  return;
}

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

// Output information about the run parameters.
// This need only be called at the beginning and end of the run.
void output_parameters()
{
  static FILE *info_;
  static time_t tStart,tFinish; // Keep track of elapsed clock time

  static int first=1;
  if(first) // At beginning of run output run parameters
  {
    sprintf(name_,"info%s",ext_);
    info_=fopen(name_,mode_);

    fprintf(info_,"--------------------------\n");
    fprintf(info_,"Model Specific Information\n");
    fprintf(info_,"--------------------------\n");
    modelinfo(info_);

    fprintf(info_,"\n--------------------------\n");
    fprintf(info_,"General Program Information\n");
    fprintf(info_,"-----------------------------\n");
    fprintf(info_,"Grid size=%d^%d\n",N,NDIMS);
    fprintf(info_,"Number of fields=%d\n",nflds);
    fprintf(info_,"L=%f\n",L);
    fprintf(info_,"dt=%f, dt/dx=%f\n",dt,dt/dx);
    if(expansion==0)
      fprintf(info_,"No expansion\n");
    else if(expansion==1)
      fprintf(info_,"Fixed background expansion\n");
    else if(expansion==2)
      fprintf(info_,"Expansion calculated self-consistently\n");
    fprintf(info_,"MPI version run with %d processors\n",numprocs);
    time(&tStart);
    fprintf(info_,"\nRun began at %s",ctime(&tStart)); // Output date in readable form
    first=0;
  }
  else // If not at beginning record elapsed time for run
  {
    time(&tFinish);
    fprintf(info_,"Run ended at %s",ctime(&tFinish)); // Output ending date
    fprintf(info_,"\nRun from t=%f to t=%f took ",t0,t);
    readable_time((int)(tFinish-tStart),info_);
    fprintf(info_,"\n");
  }

  fflush(info_);
  return;
}

// Calculate and save quantities (means, variances, etc.). If force>0 all infrequent calculations will be performed
void save(int force)
{
  static float tsave=t0; // Used to indicate when output files should be flushed.
  int flush=0; // Indicates whether to flush files, checkpoint, and perform certain infrequent calculations

  if(checkpoint_interval==0.) // In this case only checkpoint at the end of the run
    tsave=tf;

  if(t>=tsave || force>0) // Check if the program should flush
  {
    tsave = t + checkpoint_interval;
    flush=1;
  }

  if(t>0.) // Synchronize field values and derivatives. There's a special case at t=0 when fields and derivatives should stay synchronized
    evolve_fields(-.5*dt);
  if(expansion==2) // Bring a'' up to date. Leapfrog correction isn't used since all quantities are known at the time t
    evolve_scale(0);

  // Model-specific output
  if(smodel) // Allows each model to define its own output function(s) and set rescaling for other outputs
    model_output(flush,ext_);

  if(smeansvars) // Calculate means and variances of fields
    meansvars(flush);
  if(sexpansion && expansion==2) // Output scale factor and its derivatives
    scale(flush);

  // Infrequent calculations
  if(senergy && flush && t>=tenergy) // Calculate all contributions to energy density
    energy();
  if(shistograms && flush && t>=thistograms) // Calculate histograms of all output fields
    histograms();
  if(shistograms2d && flush && t>=thistograms2d) // Calculate two dimensional histograms of pairs of fields
    histograms_2d();
  if(sslices && flush && t>=tslices) // Calculate two dimensional histograms of pairs of fields
    slices();
  if(sspectra && flush && t>=tspectra) // Calculate and power spectra of all output fields
    spectra();

  if(t>0.) // Desynchronize field values and derivatives (for leapfrog)
    evolve_fields(.5*dt);
  if(scheckpoint && flush && t>=tcheckpoint) // Save an image of the grid.
    checkpoint();
}





