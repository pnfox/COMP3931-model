import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

def sum(a,b):
    
    queuedArray1 = cl.array.to_device(queue, a)
    queuedArray2 = cl.array.to_device(queue, b)
    result = cl.array.empty_like(queuedArray1)

    sumKernel = ElementwiseKernel(ctx,
            "float *queuedArray1, float *queuedArray2, float *result",
            "result[i] = queuedArray1[i] + queuedArray2[i]",
            "sumKernel")

    sumKernel(queuedArray1, queuedArray2, result)

    return result.get()

def multiply(a,b):

    queuedArray1 = cl.array.to_device(queue, a)
    queuedArray2 = cl.array.to_device(queue, b)
    result = cl.array.empty_like(queuedArray1)

    multiplyKernel = ElementwiseKernel(ctx,
            "float *a, float *b, float *result",
            "result[i] = a[i] * b[i]",
            "multiplyKernel")

    multiplyKernel(queuedArray1, queuedArray2, result)

    return result.get()

# a*b - c*d
def minusMultiples(a,b,c,d):

    queuedArray1 = cl.array.to_device(queue, a)
    queuedArray2 = cl.array.to_device(queue, b)
    queuedArray3 = cl.array.to_device(queue, c)
    queuedArray4 = cl.array.to_device(queue, d)
    result = cl.array.empty_like(queuedArray1)

    multiplyKernel = ElementwiseKernel(ctx,
            "float *a, float *b, float *c, float *d, float *result",
            "result[i] = a[i] * b[i] - c[i] * d[i]",
            "multiplyKernel")

    multiplyKernel(queuedArray1, queuedArray2, \
            queuedArray3, queuedArray4, result)

    return result.get()
