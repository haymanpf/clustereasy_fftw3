COMPILER = mpic++
FLAGS = -O3 -I${MCKENZIE_FFTW_INC_PATH} -L${MCKENZIE_FFTW_LIB_PATH}
%FLAGS = -O3 -I${MCKENZIE_FFTW_INC_PATH} -L${MCKENZIE_FFTW_LIB_PATH} -cxxlib-icc
all: latticeeasy.h model.h parameters.h latticeeasy.o evolution.o initialize.o output.o mpiutil.o ffteasy.o
	# $(COMPILER) $(FLAGS) latticeeasy.o evolution.o initialize.o output.o mpiutil.o ffteasy.o -lsrfftw_mpi -lsfftw_mpi -lsrfftw -lsfftw -lm -o latticeeasy
	$(COMPILER) $(FLAGS) latticeeasy.o evolution.o initialize.o output.o mpiutil.o ffteasy.o -lfftw3f_mpi -lfftw3f -lm -o latticeeasy

clean:
	rm -f *.o out output.txt latticeeasy *~ *.dat core*

cleaner:
	rm -f *.o out output.txt latticeeasy *~ *.dat *.img

latticeeasy.o: latticeeasy.cpp latticeeasy.h model.h parameters.h
	$(COMPILER) -c $(FLAGS) latticeeasy.cpp

evolution.o: evolution.cpp latticeeasy.h model.h parameters.h
	$(COMPILER) -c $(FLAGS) evolution.cpp

initialize.o: initialize.cpp latticeeasy.h model.h parameters.h
	$(COMPILER) -c $(FLAGS) initialize.cpp

output.o: output.cpp latticeeasy.h model.h parameters.h
	$(COMPILER) -c $(FLAGS) output.cpp

mpiutil.o: mpiutil.cpp latticeeasy.h model.h parameters.h
	$(COMPILER) -c $(FLAGS) mpiutil.cpp

ffteasy.o: ffteasy.cpp
	$(COMPILER) -c $(FLAGS) ffteasy.cpp
