#include <jni.h>
#include <iostream>
#include "cvxopt.h"
#include "edu_umich_pyongjoo_QPSolver.h"


void mat_print(PyObject* A, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << MAT_BUFD(A)[i+n*j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void data_print(jdouble* A, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A[i+n*j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


JNIEXPORT jdoubleArray JNICALL Java_edu_umich_pyongjoo_QPSolver_cvxopt_1solve_1sparse
  (JNIEnv *env, jobject jo, jdoubleArray PX, jintArray PI, jintArray PJ, jdoubleArray q, jdoubleArray GX, jintArray GI, jintArray GJ, jdoubleArray h)
{
    jsize m = env->GetArrayLength(q);
    jsize nz = env->GetArrayLength(PX);

    std::cout << "using sparse solver" << std::endl;
    std::cout << "problem size: (" << m << ")" << std::endl;

    jdouble *PXdata = env->GetDoubleArrayElements(PX, NULL);
    jint *PIdata = env->GetIntArrayElements(PI, NULL);
    jint *PJdata = env->GetIntArrayElements(PJ, NULL);

    jdouble *qdata = env->GetDoubleArrayElements(q, NULL);

    jdouble *GXdata = env->GetDoubleArrayElements(GX, NULL);
    jint *GIdata = env->GetIntArrayElements(GI, NULL);
    jint *GJdata = env->GetIntArrayElements(GJ, NULL);

    jdouble *hdata = env->GetDoubleArrayElements(h, NULL);

    // Set up cvxopt routine
    Py_Initialize();

    if (import_cvxopt() < 0) {
        fprintf(stderr, "error importing cvxopt");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers) {
        fprintf(stderr, "error importing cvxopt.solvers\n\n");
        PyErr_Print();
        PyErr_SetString(PyExc_NotImplementedError, "Not implemented");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject* options = PyObject_GetAttrString(solvers, "options");
    PyDict_SetItemString(options, "maxiters", PyLong_FromLong(50));
    PyDict_SetItemString(options, "abstol", PyFloat_FromDouble(1e-4));
    PyDict_SetItemString(options, "reltol", PyFloat_FromDouble(1e-3));
    PyObject_SetAttrString(solvers, "options", options);

    /* get reference to solvers.solvelp */
    PyObject *qp = PyObject_GetAttrString(solvers, "qp");
    if (!qp) {
        fprintf(stderr, "error referencing cvxopt.solvers.qp");
        Py_DECREF(solvers);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    /* create matrices */
    matrix *PXo = Matrix_New(nz, 1, DOUBLE);
    matrix *PIo = Matrix_New(nz, 1, INT);
    matrix *PJo = Matrix_New(m+1, 1, INT);

    PyObject *qo = (PyObject *)Matrix_New(m, 1, DOUBLE);

    matrix *GXo = Matrix_New(m, 1, DOUBLE);
    matrix *GIo = Matrix_New(m, 1, INT);
    matrix *GJo = Matrix_New(m+1, 1, INT);

    PyObject *ho = (PyObject *)Matrix_New(m, 1, DOUBLE);

    PyObject *pArgs = PyTuple_New(4);

    if (!PXo || !PIo || !PJo || !qo || !GXo || !GIo || !GJo || !ho || !pArgs) {
        fprintf(stderr, "error creating matrices");
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(PXo);
        Py_XDECREF(PIo);
        Py_XDECREF(PJo);
        Py_XDECREF(qo);
        Py_XDECREF(GXo);
        Py_XDECREF(GIo);
        Py_XDECREF(GJo);
        Py_XDECREF(ho);
        Py_XDECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    for (int i = 0; i < nz; i++) {
        MAT_BUFD(PXo)[i] = PXdata[i];
        MAT_BUFD(PIo)[i] = PIdata[i];
    }
    for (int i = 0; i < m+1; i++) {
        MAT_BUFD(PJo)[i] = PJdata[i];
    }

    for (int i = 0; i < m; i++) {
        MAT_BUFD(GXo)[i] = GXdata[i];
        MAT_BUFD(GIo)[i] = GIdata[i];
    }
    for (int i = 0; i < m+1; i++) {
        MAT_BUFD(GJo)[i] = GJdata[i];
    }

    for (int i = 0; i < m; i++) {
        MAT_BUFD(qo)[i] = qdata[i];
        MAT_BUFD(ho)[i] = hdata[i];
    }

    /* Now, we create a sparse matrix SparseP from dense PX, PI, PJ */
    PyObject *SparseP = (PyObject *)SpMatrix_NewFromIJV(PIo, PJo, PXo, m, m, DOUBLE);
    PyObject *SparseG = (PyObject *)SpMatrix_NewFromIJV(GIo, GJo, GXo, m, m, DOUBLE);


    /* pack matrices into an argument tuple - references are stolen*/
    PyTuple_SetItem(pArgs, 0, SparseP);
    PyTuple_SetItem(pArgs, 1, qo);
    PyTuple_SetItem(pArgs, 2, SparseG);
    PyTuple_SetItem(pArgs, 3, ho);

    // main routine
    PyObject *sol = PyObject_CallObject(qp, pArgs);

    if (!sol) {
        PyErr_Print();
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(PXo);
        Py_XDECREF(PIo);
        Py_XDECREF(PJo);
        Py_XDECREF(qo);
        Py_XDECREF(GXo);
        Py_XDECREF(GIo);
        Py_XDECREF(GJo);
        Py_XDECREF(ho);
        Py_DECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *x = PyDict_GetItemString(sol, "x");
    //fprintf(stdout, "\n\nx = (%5.4e, %5.4e, %5.4e, %5.4e)\n", MAT_BUFD(x)[0], MAT_BUFD(x)[1], MAT_BUFD(x)[2], MAT_BUFD(x)[3]);

    double results[m];
    for (int i = 0; i < m; i++) {
        results[i] = MAT_BUFD(x)[i];
    }

    Py_XDECREF(PXo);
    Py_XDECREF(PIo);
    Py_XDECREF(PJo);
    Py_XDECREF(qo);
    Py_XDECREF(GXo);
    Py_XDECREF(GIo);
    Py_XDECREF(GJo);
    Py_XDECREF(ho);

    Py_Finalize();

    env->ReleaseDoubleArrayElements(PX, PXdata, JNI_ABORT);
    env->ReleaseIntArrayElements(PI, PIdata, JNI_ABORT);
    env->ReleaseIntArrayElements(PJ, PJdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(q, qdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(GX, GXdata, JNI_ABORT);
    env->ReleaseIntArrayElements(GI, GIdata, JNI_ABORT);
    env->ReleaseIntArrayElements(GJ, GJdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(h, hdata, JNI_ABORT);

    // output

    jdoubleArray output = env->NewDoubleArray( m );
    env->SetDoubleArrayRegion(output, 0, m, &results[0] );

    return output;
}


JNIEXPORT jdoubleArray JNICALL Java_edu_umich_pyongjoo_QPSolver_cvxopt_1solve___3D_3D_3D_3D
  (JNIEnv *env, jobject jo, jdoubleArray P, jdoubleArray q, jdoubleArray G, jdoubleArray h)
{
    jsize m = env->GetArrayLength(q);

    std::cout << "problem size: (" << m << ")" << std::endl;

    jdouble *Pdata = env->GetDoubleArrayElements(P, NULL);
    jdouble *qdata = env->GetDoubleArrayElements(q, NULL);
    jdouble *Gdata = env->GetDoubleArrayElements(G, NULL);
    jdouble *hdata = env->GetDoubleArrayElements(h, NULL);

    // Set up cvxopt routine
    Py_Initialize();

    if (import_cvxopt() < 0) {
        fprintf(stderr, "error importing cvxopt");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers) {
        fprintf(stderr, "error importing cvxopt.solvers\n\n");
        PyErr_Print();
        PyErr_SetString(PyExc_NotImplementedError, "Not implemented");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject* options = PyObject_GetAttrString(solvers, "options");
    PyDict_SetItemString(options, "maxiters", PyLong_FromLong(50));
    PyDict_SetItemString(options, "abstol", PyFloat_FromDouble(1e-4));
    PyDict_SetItemString(options, "reltol", PyFloat_FromDouble(1e-3));
    PyObject_SetAttrString(solvers, "options", options);

    /* get reference to solvers.solvelp */
    PyObject *qp = PyObject_GetAttrString(solvers, "qp");
    if (!qp) {
        fprintf(stderr, "error referencing cvxopt.solvers.qp");
        Py_DECREF(solvers);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    /* create matrices */
    PyObject *Po = (PyObject *)Matrix_New(m, m, DOUBLE);
    PyObject *qo = (PyObject *)Matrix_New(m, 1, DOUBLE);

    PyObject *Go = (PyObject *)Matrix_New(m, m, DOUBLE);
    PyObject *ho = (PyObject *)Matrix_New(m, 1, DOUBLE);

    PyObject *pArgs = PyTuple_New(4);

    if (!Po || !qo || !Go || !ho || !pArgs) {
        fprintf(stderr, "error creating matrices");
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(Po);
        Py_XDECREF(qo);
        Py_XDECREF(Go);
        Py_XDECREF(ho);
        Py_XDECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    for (int i = 0; i < m*m; i++) {
        MAT_BUFD(Po)[i] = Pdata[i];
        MAT_BUFD(Go)[i] = Gdata[i];
    }

    for (int i = 0; i < m; i++) {
        MAT_BUFD(qo)[i] = qdata[i];
        MAT_BUFD(ho)[i] = hdata[i];
    }


    /* pack matrices into an argument tuple - references are stolen*/
    PyTuple_SetItem(pArgs, 0, Po);
    PyTuple_SetItem(pArgs, 1, qo);
    PyTuple_SetItem(pArgs, 2, Go);
    PyTuple_SetItem(pArgs, 3, ho);

    // main routine
    PyObject *sol = PyObject_CallObject(qp, pArgs);

    if (!sol) {
        PyErr_Print();
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(Po);
        Py_XDECREF(qo);
        Py_XDECREF(Go);
        Py_XDECREF(ho);
        Py_DECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *x = PyDict_GetItemString(sol, "x");
    //fprintf(stdout, "\n\nx = (%5.4e, %5.4e, %5.4e, %5.4e)\n", MAT_BUFD(x)[0], MAT_BUFD(x)[1], MAT_BUFD(x)[2], MAT_BUFD(x)[3]);

    double results[m];
    for (int i = 0; i < m; i++) {
        results[i] = MAT_BUFD(x)[i];
    }

    Py_XDECREF(Po);
    Py_XDECREF(qo);
    Py_XDECREF(Go);
    Py_XDECREF(ho);

    Py_Finalize();

    env->ReleaseDoubleArrayElements(P, Pdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(q, qdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(G, Gdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(h, hdata, JNI_ABORT);

    // output

    jdoubleArray output = env->NewDoubleArray( m );
    env->SetDoubleArrayRegion(output, 0, m, &results[0] );

    return output;
}



JNIEXPORT jdoubleArray JNICALL Java_edu_umich_pyongjoo_QPSolver_cvxopt_1solve___3D_3D_3D_3D_3D_3D
  (JNIEnv *env, jobject jo,
   jdoubleArray P, jdoubleArray q, jdoubleArray G, jdoubleArray h, jdoubleArray A, jdoubleArray b)
{
	// input
    // Q: (m, m)
    // p: (m, 1)
    // G: (m, m)
    // h: (m, 1)
    // A: (n, m)
    // b: (n, 1)
    jsize m = env->GetArrayLength(q);
    jsize n = env->GetArrayLength(b);

    std::cout << "problem size: (" << n << ", " << m << ")" << std::endl;

    jdouble *Pdata = env->GetDoubleArrayElements(P, NULL);
    jdouble *qdata = env->GetDoubleArrayElements(q, NULL);
    jdouble *Gdata = env->GetDoubleArrayElements(G, NULL);
    jdouble *hdata = env->GetDoubleArrayElements(h, NULL);
    jdouble *Adata = env->GetDoubleArrayElements(A, NULL);
    jdouble *bdata = env->GetDoubleArrayElements(b, NULL);


    // Set up cvxopt routine
    Py_Initialize();

    if (import_cvxopt() < 0) {
        fprintf(stderr, "error importing cvxopt");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers) {
        fprintf(stderr, "error importing cvxopt.solvers");
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject* options = PyObject_GetAttrString(solvers, "options");
    PyDict_SetItemString(options, "maxiters", PyLong_FromLong(50));
    PyObject_SetAttrString(solvers, "options", options);

    /* get reference to solvers.solvelp */
    PyObject *qp = PyObject_GetAttrString(solvers, "qp");
    if (!qp) {
        fprintf(stderr, "error referencing cvxopt.solvers.qp");
        Py_DECREF(solvers);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    /* create matrices */
    PyObject *Po = (PyObject *)Matrix_New(m, m, DOUBLE);
    PyObject *qo = (PyObject *)Matrix_New(m, 1, DOUBLE);

    PyObject *Go = (PyObject *)Matrix_New(m, m, DOUBLE);
    PyObject *ho = (PyObject *)Matrix_New(m, 1, DOUBLE);

    PyObject *Ao = (PyObject *)Matrix_New(n, m, DOUBLE);
    PyObject *bo = (PyObject *)Matrix_New(n, 1, DOUBLE);
    PyObject *pArgs = PyTuple_New(6);

    if (!Po || !qo || !Go || !ho || !Ao || !bo || !pArgs) {
        fprintf(stderr, "error creating matrices");
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(Po);
        Py_XDECREF(qo);
        Py_XDECREF(Go);
        Py_XDECREF(ho);
        Py_XDECREF(Ao);
        Py_XDECREF(bo);
        Py_XDECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    for (int i = 0; i < m*m; i++) {
        MAT_BUFD(Po)[i] = Pdata[i];
        MAT_BUFD(Go)[i] = Gdata[i];
    }

    for (int i = 0; i < m; i++) {
        MAT_BUFD(qo)[i] = qdata[i];
        MAT_BUFD(ho)[i] = hdata[i];
    }

    for (int i = 0; i < n*m; i++) {
        MAT_BUFD(Ao)[i] = Adata[i];
    }

    for (int i = 0; i < n; i++) {
        MAT_BUFD(bo)[i] = bdata[i];
    }

    //data_print(Pdata, m, m);
    //data_print(qdata, m, 1);
    //data_print(Gdata, m, m);
    //data_print(hdata, m, 1);
    //data_print(Adata, n, m);
    //data_print(bdata, n, 1);

    //mat_print(Po, m, m);
    //mat_print(qo, m, 1);
    //mat_print(Go, m, m);
    //mat_print(ho, m, 1);
    //mat_print(Ao, n, m);
    //mat_print(bo, n, 1);


    /* pack matrices into an argument tuple - references are stolen*/
    PyTuple_SetItem(pArgs, 0, Po);
    PyTuple_SetItem(pArgs, 1, qo);
    PyTuple_SetItem(pArgs, 2, Go);
    PyTuple_SetItem(pArgs, 3, ho);
    PyTuple_SetItem(pArgs, 4, Ao);
    PyTuple_SetItem(pArgs, 5, bo);

    // main routine
    PyObject *sol = PyObject_CallObject(qp, pArgs);

    if (!sol) {
        PyErr_Print();
        Py_DECREF(solvers);
        Py_DECREF(qp);
        Py_XDECREF(Po);
        Py_XDECREF(qo);
        Py_XDECREF(Go);
        Py_XDECREF(ho);
        Py_XDECREF(Ao);
        Py_XDECREF(bo);
        Py_DECREF(pArgs);
        jdoubleArray output = env->NewDoubleArray(0);
        return output;
    }

    PyObject *x = PyDict_GetItemString(sol, "x");
    //fprintf(stdout, "\n\nx = (%5.4e, %5.4e, %5.4e, %5.4e)\n", MAT_BUFD(x)[0], MAT_BUFD(x)[1], MAT_BUFD(x)[2], MAT_BUFD(x)[3]);

    double results[m];
    for (int i = 0; i < m; i++) {
        results[i] = MAT_BUFD(x)[i];
    }

    //Py_DECREF(solvers);
    //Py_DECREF(qp);
    //Py_DECREF(pArgs);
    //Py_DECREF(sol);
    Py_XDECREF(Po);
    Py_XDECREF(qo);
    Py_XDECREF(Go);
    Py_XDECREF(ho);
    Py_XDECREF(Ao);
    Py_XDECREF(bo);

    Py_Finalize();

    env->ReleaseDoubleArrayElements(P, Pdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(q, qdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(G, Gdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(h, hdata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(A, Adata, JNI_ABORT);
    env->ReleaseDoubleArrayElements(b, bdata, JNI_ABORT);

    // output

    jdoubleArray output = env->NewDoubleArray( m );
    env->SetDoubleArrayRegion(output, 0, m, &results[0] );

    return output;
}




