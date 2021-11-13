# This script works on Mac
# python-config --includes
g++ -I/System/Library/Frameworks/JavaVM.framework/Headers -Icvxopt/src/C \
    -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
    -c edu_umich_pyongjoo_QPSolver.cpp
# python-config --ldflags
g++ -dynamiclib \
    -L/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config \
    -lpython2.7 -o libqpsolver.jnilib edu_umich_pyongjoo_QPSolver.o

