# what flags you want to pass to the C compiler & linker
AM_CPPFLAGS = -DWITH_MPI  -Wall -Wlong-long  -O3 $(MPI_CXXFLAGS)      # --pedantic
AM_LDFLAGS =  -lgsl -lgslcblas

#CXX = $(MPICXX)
CXX = mpic++

bin_PROGRAMS = cocoa

cocoa_SOURCES = cocoa.cpp
cocoa_LDFLAGS = $(MPI_CXXLDFLAGS)    -fopenmp
cocoa_LDADD = -lboost_system -lboost_timer -lboost_chrono -lrt  -lboost_mpi -lboost_serialization   -lgsl -lgslcblas -lm -lboost_thread


