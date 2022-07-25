cimport cython
from cython.parallel import prange
from libc.math cimport sin, cos, floor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef unsigned char getRectSubPix(const unsigned char[:, ::1] image, const unsigned int h, const unsigned int w,
                                 const float x, const float y) nogil:
    cdef int x0 = <int> floor(x)
    cdef int x1 = x0 + 1
    cdef int y0 = <int> floor(y)
    cdef int y1 = y0 + 1

    # Clipping
    if x0 < 0:
        x0 = 0
    elif x0 >= <int> w:
        x0 = w - 1
    if x1 < 0:
        x1 = 0
    elif x1 >= <int> w:
        x1 = w - 1
    if y0 < 0:
        y0 = 0
    elif y0 >= <int> h:
        y0 = h - 1
    if y1 < 0:
        y1 = 0
    elif y1 >= <int> h:
        y1 = h - 1

    cdef unsigned char i_a = image[y0, x0]
    cdef unsigned char i_b = image[y1, x0]
    cdef unsigned char i_c = image[y0, x1]
    cdef unsigned char i_d = image[y1, x1]

    cdef float w_a = (x1 - x) * (y1 - y)
    cdef float w_b = (x1 - x) * (y - y0)
    cdef float w_c = (x - x0) * (y1 - y)
    cdef float w_d = (x - x0) * (y - y0)

    cdef float value = w_a * i_a + w_b * i_b + w_c * i_c + w_d * i_d
    if value < 0:
        value = 0
    elif value > 255:
        value = 255

    return <unsigned char> value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef unsigned char getRectSubPix_channel(const unsigned char[:, :, ::1] image, const unsigned int h,
                                         const unsigned int w, const Py_ssize_t channel, const float x,
                                         const float y) nogil:
    cdef int x0 = <int> floor(x)
    cdef int x1 = x0 + 1
    cdef int y0 = <int> floor(y)
    cdef int y1 = y0 + 1

    # Clipping
    if x0 < 0:
        x0 = 0
    elif x0 >= <int> w:
        x0 = w - 1
    if x1 < 0:
        x1 = 0
    elif x1 >= <int> w:
        x1 = w - 1
    if y0 < 0:
        y0 = 0
    elif y0 >= <int> h:
        y0 = h - 1
    if y1 < 0:
        y1 = 0
    elif y1 >= <int> h:
        y1 = h - 1

    cdef unsigned char i_a = image[y0, x0, channel]
    cdef unsigned char i_b = image[y1, x0, channel]
    cdef unsigned char i_c = image[y0, x1, channel]
    cdef unsigned char i_d = image[y1, x1, channel]

    cdef float w_a = (x1 - x) * (y1 - y)
    cdef float w_b = (x1 - x) * (y - y0)
    cdef float w_c = (x - x0) * (y1 - y)
    cdef float w_d = (x - x0) * (y - y0)

    cdef float value = w_a * i_a + w_b * i_b + w_c * i_c + w_d * i_d
    if value < 0:
        value = 0
    elif value > 255:
        value = 255

    return <unsigned char> value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def unwrap(const unsigned char[:, ::1] image, const unsigned int h, const unsigned int w,
                        unsigned char[:, ::1] tread, const unsigned int tread_height,
                        const unsigned int tread_width, const int cx1, const int cy1, const float a1, const float b1,
                        float theta1, const int cx2, const int cy2, const float a2, const float b2, float theta2,
                        float start, float end):

    # Convert to radians
    start = start * 3.141592 / 180.0
    end = end * 3.141592 / 180.0
    theta1 = theta1 * 3.141592 / 180.0
    theta2 = theta2 * 3.141592 / 180.0

    # Declare used variables
    cdef float x1, x2, y1, y2, t, x_step, y_step, x_origin, y_origin
    # Ellipse step (in radians)
    cdef float step = (end - start) / (tread_height - 1)
    # Loop variables
    cdef Py_ssize_t x, y
    # Multithreaded loop
    for y in prange(tread_height, nogil=True):
        # Current position
        t = start + step * y
        # Points on ellipse1 and ellipse2
        x1 = a1 * cos(t) * cos(theta1) - b1 * sin(t) * sin(theta1) + cx1
        y1 = a1 * cos(t) * sin(theta1) + b1 * sin(t) * cos(theta1) + cy1
        x2 = a2 * cos(t) * cos(theta2) - b2 * sin(t) * sin(theta2) + cx2
        y2 = a2 * cos(t) * sin(theta2) + b2 * sin(t) * cos(theta2) + cy2
        # We will be stepping over a line created between them
        x_step = (x2 - x1) / tread_width
        y_step = (y2 - y1) / tread_width
        for x in range(tread_width):
            x_origin = x1 + x * x_step
            y_origin = y1 + x * y_step
            # Bilinear interpolation
            tread[y, x] = getRectSubPix(image, h, w, x_origin, y_origin)

    return tread


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def unwrap_multichannel(const unsigned char[:, :, ::1] image, const unsigned int h, const unsigned int w,
                        const unsigned int channels, unsigned char[:, :, ::1] tread, const unsigned int tread_height,
                        const unsigned int tread_width, const int cx1, const int cy1, const float a1, const float b1,
                        float theta1, const int cx2, const int cy2, const float a2, const float b2, float theta2,
                        float start, float end):

    # Convert to radians
    start = start * 3.141592 / 180.0
    end = end * 3.141592 / 180.0
    theta1 = theta1 * 3.141592 / 180.0
    theta2 = theta2 * 3.141592 / 180.0

    # Declare used variables
    cdef float x1, x2, y1, y2, t, x_step, y_step, x_origin, y_origin
    # Ellipse step (in radians)
    cdef float step = (end - start) / (tread_height - 1)
    # Loop variables
    cdef Py_ssize_t x, y, c
    # Multithreaded loop
    for y in prange(tread_height, nogil=True):
        # Current position
        t = start + step * y
        # Points on ellipse1 and ellipse2
        x1 = a1 * cos(t) * cos(theta1) - b1 * sin(t) * sin(theta1) + cx1
        y1 = a1 * cos(t) * sin(theta1) + b1 * sin(t) * cos(theta1) + cy1
        x2 = a2 * cos(t) * cos(theta2) - b2 * sin(t) * sin(theta2) + cx2
        y2 = a2 * cos(t) * sin(theta2) + b2 * sin(t) * cos(theta2) + cy2
        # We will be stepping over a line created between them
        x_step = (x2 - x1) / tread_width
        y_step = (y2 - y1) / tread_width
        for x in range(tread_width):
            x_origin = x1 + x * x_step
            y_origin = y1 + x * y_step
            # Bilinear interpolation
            for c in range(channels):
                tread[y, x, c] = getRectSubPix_channel(image, h, w, c, x_origin, y_origin)

    return tread