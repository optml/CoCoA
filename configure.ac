#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT(cocoa, 1.0, takac.mt@gmail.com)
AC_MSG_NOTICE([CoCoA Framework.])
AM_INIT_AUTOMAKE([1.12 foreign no-define])




 
echo "                             Testing for a C compiler"
AC_PROG_CC
echo "                             Testing for a C++ compiler"
AC_PROG_CXX
AC_LANG(C++)




# Checks for programs.
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# Checks for libraries.

echo " "
echo "                             Testing for GSL and CLBAS Libraries "
echo " "

AC_CHECK_LIB([gslcblas],[cblas_dgemm])
if test "$ac_cv_lib_gslcblas_cblas_dgemm" == no
then
	AC_MSG_ERROR([Error, the boost_system header is missing!])
fi





#-lboost_system -lboost_timer -lboost_chrono -lrt  -lboost_mpi -lboost_serialization   -lgsl -lgslcblas -lm -lboost_thread
echo " "
echo "                             Testing for Boost Libraries "
echo " "
AC_CHECK_LIB(boost_system)






# Checks for header files.
AC_CHECK_HEADERS(iostream)
AC_CHECK_HEADERS(stdio.h)
AC_CHECK_HEADERS(stdlib.h)
AC_CHECK_HEADERS(unistd.h)
# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Configure MPI
AC_PROG_CXX([mpic++ mpicxx])



AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT
