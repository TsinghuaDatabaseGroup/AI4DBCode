/*
 * Cone Programming Test
 *
 */


#include "cvxopt.h"


int main(int argc, char *argv[]) {

  Py_Initialize();

  if (import_cvxopt() < 0) {
    fprintf(stderr, "error importing cvxopt");
    return 1;
  }

  /* import cvxopt.solvers */
  PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
  if (!solvers) {
    fprintf(stderr, "error importing cvxopt.solvers");
    return 1;
  }

  /* get reference to solvers.solvelp */
  PyObject *cp = PyObject_GetAttrString(solvers, "cp");
  if (!cp) {
    fprintf(stderr, "error referencing cvxopt.solvers.cp");
    Py_DECREF(solvers);
    return 1;
  }

  /* create matrices */
  PyObject *P = (PyObject *)Matrix_New(4, 4, DOUBLE);
  PyObject *q = (PyObject *)Matrix_New(4, 1, DOUBLE);

  PyObject *G = (PyObject *)Matrix_New(4, 4, DOUBLE);
  PyObject *h = (PyObject *)Matrix_New(4, 1, DOUBLE);

  PyObject *dim = PyDict_New();
  PyObject *diml = PyFloat_FromDouble(4);
  PyObject *dimq = PyList_New(0);
  PyObject *dims = PyList_New(0);
  PyDict_SetItemString(dim, "l", diml);
  PyDict_SetItemString(dim, "q", dimq);
  PyDict_SetItemString(dim, "s", dims);

  PyObject *A = (PyObject *)Matrix_New(3, 4, DOUBLE);
  PyObject *b = (PyObject *)Matrix_New(3, 1, DOUBLE);
  PyObject *pArgs = PyTuple_New(7);
  if (!P || !q || !G || !h || !A || !b || !pArgs) {
    fprintf(stderr, "error creating matrices");
    Py_DECREF(solvers);
    Py_DECREF(cp);
    Py_XDECREF(P);
    Py_XDECREF(q);
    Py_XDECREF(G);
    Py_XDECREF(h);
    Py_XDECREF(A);
    Py_XDECREF(b);
    Py_DECREF(diml);
    Py_DECREF(dimq);
    Py_DECREF(dims);
    Py_DECREF(dim);
    Py_XDECREF(pArgs);
    return 1;
  }

  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
          MAT_BUFD(P)[i+j*4] = (double) (i == j);
          MAT_BUFD(G)[i+j*4] = -1.0 * ((double) (i == j));
      }
      MAT_BUFD(q)[i] = 0.0;
      MAT_BUFD(h)[i] = 0.0;
  }

  MAT_BUFD(A)[0] = 1.0;
  MAT_BUFD(A)[1] = 0.0;
  MAT_BUFD(A)[2] = 0.0;
  MAT_BUFD(A)[3] = 1.0;
  MAT_BUFD(A)[4] = 1.0;
  MAT_BUFD(A)[5] = 0.0;
  MAT_BUFD(A)[6] = 1.0;
  MAT_BUFD(A)[7] = 0.0;
  MAT_BUFD(A)[8] = 1.0;
  MAT_BUFD(A)[9] = 1.0;
  MAT_BUFD(A)[10] = 1.0;
  MAT_BUFD(A)[11] = 1.0;

  MAT_BUFD(b)[0] = 1.0;
  MAT_BUFD(b)[1] = 0.8;
  MAT_BUFD(b)[2] = 0.3;

  /* pack matrices into an argument tuple - references are stolen*/
  PyTuple_SetItem(pArgs, 0, P);
  PyTuple_SetItem(pArgs, 1, q);
  PyTuple_SetItem(pArgs, 2, G);
  PyTuple_SetItem(pArgs, 3, h);
  PyTuple_SetItem(pArgs, 4, A);
  PyTuple_SetItem(pArgs, 5, b);

  PyObject *sol = PyObject_CallObject(cp, pArgs);
  if (!sol) {
    PyErr_Print();
    Py_DECREF(solvers);
    Py_DECREF(cp);
    Py_DECREF(pArgs);
    return 1;
  }

  PyObject *x = PyDict_GetItemString(sol, "x");
  fprintf(stdout, "\n\nx = (%5.4e, %5.4e, %5.4e, %5.4e)\n", MAT_BUFD(x)[0], MAT_BUFD(x)[1], MAT_BUFD(x)[2], MAT_BUFD(x)[3]);

  Py_DECREF(solvers);
  Py_DECREF(cp);
  Py_DECREF(pArgs);
  Py_DECREF(sol);

  Py_Finalize();
  return 0;
}

