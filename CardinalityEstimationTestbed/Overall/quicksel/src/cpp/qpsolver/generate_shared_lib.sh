JAVA_HOME=/usr/lib/jvm/java-8-oracle
#JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
#JAVA_INCLUDE_MAC=/System/Library/Frameworks/JavaVM.framework/Headers
GCC_FLAG="-fPIC -shared"
SHARED_LIB_FLAG=-shared
#SHARED_LIB_FLAG_MAC=-dynamiclib
LIB_OUTPUT=libqpsolver.so
#LIB_OUTPUT_MAC=libqpsolver.jnilib

g++ -I${JAVA_HOME}/include/ -I${JAVA_HOME}/include/linux/ -Icvxopt/src/C \
    $(python2.7-config --includes) ${GCC_FLAG} \
    -c edu_umich_pyongjoo_QPSolver.cpp
g++ ${GCC_FLAG} -Wl,-no-undefined,-export-dynamic $(python2.7-config --libs) \
    -o ${LIB_OUTPUT} edu_umich_pyongjoo_QPSolver.o \
    $(python2.7-config --ldflags)

