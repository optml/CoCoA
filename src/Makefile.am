# what flags you want to pass to the C compiler & linker
AM_CPPFLAGS = -DWITH_MPI --pedantic -Wall -std=c99 -O3 $(MPI_CXXFLAGS)
AM_LDFLAGS =

#CXX = $(MPICXX)
CXX = mpic++

bin_PROGRAMS = cocoa

cocoa_SOURCES = cocoa.cpp
cocoa_LDFLAGS = $(MPI_CXXLDFLAGS)



