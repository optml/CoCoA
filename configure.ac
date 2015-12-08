#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT(cocoa, 1.0, takac.mt@gmail.com)
AM_INIT_AUTOMAKE([1.12 foreign no-define])
AC_MSG_NOTICE([CoCoA Framework.])

#AC_CONFIG_SRCDIR([my_config.h])
#AC_CONFIG_HEADERS([config.h])

 
echo "                             Testing for a C compiler"
AC_PROG_CC
echo "                             Testing for a C++ compiler"
AC_PROG_CXX
AC_LANG(C++)

AC_CHECK_HEADERS(iostream)


# Checks for programs.
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Configure MPI


AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT
