#include <Python.h>
#include <emmintrin.h>  // Include for SSE2 intrinsics

static float cosine_distance_sse(const float* a, const float* b, size_t n) {
    __m128 sum_ab = _mm_setzero_ps(), sum_a2 = _mm_setzero_ps(), sum_b2 = _mm_setzero_ps();
    for (size_t i = 0; i < n; i += 4) {
        __m128 vec_a = _mm_load_ps(&a[i]);
        __m128 vec_b = _mm_load_ps(&b[i]);
        sum_ab = _mm_add_ps(sum_ab, _mm_mul_ps(vec_a, vec_b));
        sum_a2 = _mm_add_ps(sum_a2, _mm_mul_ps(vec_a, vec_a));
        sum_b2 = _mm_add_ps(sum_b2, _mm_mul_ps(vec_b, vec_b));
    }

    // Horizontal add for sum_ab, sum_a2, sum_b2
    sum_ab = _mm_hadd_ps(sum_ab, sum_ab);
    sum_ab = _mm_hadd_ps(sum_ab, sum_ab);
    sum_a2 = _mm_hadd_ps(sum_a2, sum_a2);
    sum_a2 = _mm_hadd_ps(sum_a2, sum_a2);
    sum_b2 = _mm_hadd_ps(sum_b2, sum_b2);
    sum_b2 = _mm_hadd_ps(sum_b2, sum_b2);

    float ab, a2, b2;
    _mm_store_ss(&ab, sum_ab);
    _mm_store_ss(&a2, sum_a2);
    _mm_store_ss(&b2, sum_b2);

    return ab / (sqrtf(a2) * sqrtf(b2));
}

static PyObject* py_cosine_distance(PyObject* self, PyObject* args) {
    PyObject *py_a, *py_b;
    if (!PyArg_ParseTuple(args, "OO", &py_a, &py_b)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(py_a, NPY_FLOAT, NPY_IN_ARRAY);
    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(py_b, NPY_FLOAT, NPY_IN_ARRAY);

    if (!a_array || !b_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    float *a = (float*)PyArray_DATA(a_array);
    float *b = (float*)PyArray_DATA(b_array);
    size_t n = PyArray_SIZE(a_array);

    float result = cosine_distance_sse(a, b, n);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return PyFloat_FromDouble((double)result);
}

static PyMethodDef CosineMethods[] = {
    {"cosine_distance", py_cosine_distance, METH_VARARGS, "Calculate the cosine distance using SSE."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cosinemodule = {
    PyModuleDef_HEAD_INIT,
    "cosine_distance",
    NULL,
    -1,
    CosineMethods
};

PyMODINIT_FUNC PyInit_cosine_distance(void) {
    import_array();
    return PyModule_Create(&cosinemodule);
}
