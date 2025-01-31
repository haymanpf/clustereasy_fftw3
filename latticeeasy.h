/*
This file contains the global variable declarations, function declarations, 
and some definitions used in many of the routines. The global variables are 
defined in the file latticeeasy.cpp.
*/

#ifndef _LATTICEEASYHEADER_
#define _LATTICEEASYHEADER_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
// #include <srfftw_mpi.h>
// #include <sfftw_mpi.h>
#include <fftw3-mpi.h> // PH

const float pi = (float)(2.*asin(1.));
inline float pw2(float x) {return x*x;} // Useful macro for squaring floats

/////////////////////////////////INCLUDE ADJUSTABLE PARAMETERS///////////////////
#include "parameters.h"

/////////////////////////////////GLOBAL DYNAMIC VARIABLES////////////////////////
extern float t,t0; // Current time and initial time (t0=0 unless the run is a continuation of a previous one)
extern float a,ad,ad2,aterm; // Scale factor and its derivatives (aterm is a combination of the others used in the equations of motion)
extern float hubble_init; // Initial value of the Hubble constant
extern int run_number; // 0 for a first run, 1 for a continuation of a "0" run, etc.. Stored in the grid image (see checkpoint() function).
extern int no_initialization; // If this variable is set to 1 by the model file then the fields will not be initialized in the normal way.
extern float rescaling; // Rescaling for output. This is left as 1 unless the model file modifies it.
extern char ext_[500]; // Extension for filenames - set once and used by all functions
extern int nfldsout; // Number of fields to output
extern char mode_[]; // Mode in which to open files, i.e. write ("w") or append ("a+"). Depends on the variable continue_run and on whether a previous grid image was found.
// Variables specific to MPI version
extern int my_rank, numprocs; // Rank of current process and total number of processors (p is a loop index for processors)
extern int n; // Size of the first dimension of the array at each processor
extern int my_start_position;  // start position in the array for each processor.  my_start_position will equal my_rank * n if all processors are allotted equal portions n of the array. 
extern float *fp, *fdp; // Pointers to the beginning of the field and field derivative arrays
extern float *fstore,*fdstore; // If grid images or full-array slices are being stored the root processor needs an extra array to gather data from other processors
extern int fstore_size,fdstore_size; // Size of the fstore array
extern int buffers_up_to_date; // Keeps track of whether the buffers are up to date. Should be set to zero whenever they are taken out of synch and checked whenever they are needed.

/////////////////////////////////NON-ADJUSTABLE VARIABLES////////////////////////
const float dx=L/(float)N; // Distance between adjacent gridpoints

/////////////////////////////////DIMENSIONAL SPECIFICATIONS//////////////////////
#if NDIMS==1
extern float **f,**fd; // Field values and derivatives
const int gridsize=N; // Number of spatial points in the grid
#define FIELD(fld) f[fld][i]
#define FIELDD(fld) fd[fld][i]
#define FIELDPOINT(fld,i,j,k) f[fld][k]
#define FIELDPOINTD(fld,i,j,k) fd[fld][k]
#define LOOP for(i=1;i<=n;i++)
#define INDEXLIST int i, ...
#define DECLARE_INDICES int i;
#elif NDIMS==2
extern float ***f,***fd; // Field values and derivatives
const int gridsize=N*N; // Number of spatial points in the grid
#define FIELD(fld) f[fld][i][j]
#define FIELDD(fld) fd[fld][i][j]
#define FIELDPOINT(fld,i,j,k) f[fld][j][k]
#define FIELDPOINTD(fld,i,j,k) fd[fld][j][k]
#define LOOP for(i=1;i<=n;i++) for(j=0;j<N;j++)
#define INDEXLIST int i, int j, ...
#define DECLARE_INDICES int i,j;
#elif NDIMS==3
extern float ****f,****fd; // Field values and derivatives
const int gridsize=N*N*N; // Number of spatial points in the grid
#define FIELD(fld) f[fld][i][j][k]
#define FIELDD(fld) fd[fld][i][j][k]
#define FIELDPOINT(fld,i,j,k) f[fld][i][j][k]
#define FIELDPOINTD(fld,i,j,k) fd[fld][i][j][k]
#define LOOP for(i=1;i<=n;i++) for(j=0;j<N;j++) for(k=0;k<N;k++)
#define INDEXLIST int i, int j, int k
#define DECLARE_INDICES int i,j,k;
#endif

/////////////////////////////////INCLUDE SPECIFIC MODEL//////////////////////////
#include "model.h"

/////////////////////////////////FUNCTION DECLARATIONS///////////////////////////
// initialize.cpp
void initialize(); // Set initial parameters and field values
// evolution.cpp
float gradient_energy(int fld); // Calculate the gradient energy, <|Grad(f)|^2>=<-f Lapl(f)>, of a field
float laplb(int fld, INDEXLIST); // LSR -- Laplacian on the boundary
//float lapls(int fld, INDEXLIST); // LSR -- Laplacian for scale()
void evolve_scale(float d); // Calculate the scale factor and its derivatives
void evolve_fields(float d); // Advance the field values and scale factor using the first derivatives
void evolve_derivs(float d); // Calculate second derivatives of fields and use them to advance first derivatives. Also calls evolve_scale().
// output.cpp
void output_parameters(); // Output information about the run parameters
void save(int force); // Calculate and save quantities (means, variances, spectra, etc.)
// ffteasy.cpp
void fftr1(float f[], int N, int forward, int my_rank, int numprocs); // Do a Fourier transform of a 1D array of real numbers. Used when NDIMS=1.
// mpiutil.cpp
void cast_field_arrays(int array_size); // Allocate the memory for the field and derivative arrays
void free_field_arrays(); // Delete the memory allocated for the field and derivative arrays
void update_buffers(); // Copy edge information between processors

#endif // End of conditional for definition of _LATTICEEASYHEADER_ macro






