import tensorflow as tf
from tensorflow import cos, sin
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam

"""
This demo consists of an approach to surface reconstruction via differentiable rendering. The idea is to learn the
parameters defining a quadratic surface from the pixel intensities captured by a virtual camera. The parameters of the
surface are related to the pixel intensities via the Phong reflection model.

Author: Christopher Strohmeier
Date: August 2023

"""


# === Camera Configuration ===

WIDTH, HEIGHT = tf.constant(60., dtype=tf.float32), tf.constant(40., dtype=tf.float32)
MEAN = tf.sqrt(WIDTH * HEIGHT)
h_bar = tf.constant(1e-3, dtype=tf.float32)
width = h_bar * (WIDTH / MEAN)
height = h_bar * (HEIGHT / MEAN)
pixel_length = h_bar / MEAN
WIDTH, HEIGHT = tf.cast(WIDTH, dtype=tf.int32), tf.cast(HEIGHT, dtype=tf.int32)

ZOOM_PARAM_MIN = tf.constant(.5, dtype=tf.float32)
ZOOM_PARAM_MEAN = tf.constant(1., dtype=tf.float32)
ZOOM_PARAM_MAX = tf.constant(2., dtype=tf.float32)
zoom_param = tf.constant(ZOOM_PARAM_MEAN, dtype=tf.float32)
zoom = h_bar * zoom_param

pi = tf.constant(3.1415926, dtype=tf.float32)
the = tf.constant(pi, dtype=tf.float32)
phi = tf.constant(0., dtype=tf.float32)
FRONT = tf.stack([cos(the) * cos(phi), sin(the) * cos(phi), sin(phi)])
RIGHT = tf.stack([sin(the), -cos(the), 0.])
UP = tf.linalg.cross(RIGHT, FRONT)
FRONT = tf.reshape(FRONT, shape=(3, 1))
RIGHT = tf.reshape(RIGHT, shape=(3, 1))
UP = tf.reshape(UP, shape=(3, 1))

EYE = tf.constant([5., 0., 0.], dtype=tf.float32)
EYE = tf.reshape(EYE, shape=(3, 1))

# === Light Configuration ===

i_a = tf.constant(.1, dtype=tf.float32)
shininess = tf.constant(16., dtype=tf.float32)

LIGHT = tf.constant([6., 0., 0.], dtype=tf.float32)
LIGHT = tf.reshape(LIGHT, shape=(3, 1))
i_d = tf.constant(.5, dtype=tf.float32)
i_s = tf.constant(.5, dtype=tf.float32)

k_a = tf.constant(.4, dtype=tf.float32)
k_d = tf.constant(.4, dtype=tf.float32)
k_s = tf.constant(.4, dtype=tf.float32)


# === Primary Section ===

@tf.function
def get_image(a00, a01, a02, a11, a12, a22, b0, b1, b2, image):
    """
    Input parameters define a quadratic surface via the equation q(x) = 1, where q(x) = (1 / 2) x.T A x - b.T x and
    A is a symmetric matrix with the given entries.

    get_image will use the camera and light parameters to take a "snapshot" of the quadratic surface q(x) = 1

    :param image (tf.Variable): image to be updated
    :return: updated image
    """

    # Care must be taken to ensure that tensorflow recognizes our output as a differentiable function of the inputs
    row_0 = tf.stack([a00, a01, a02])
    row_1 = tf.stack([a01, a11, a12])
    row_2 = tf.stack([a02, a12, a22])
    A = tf.stack([row_0, row_1, row_2])
    b = tf.stack([b0, b1, b2])
    b = tf.reshape(b, shape=(3, 1))

    # Iterating over each pixel in the image
    for i in tf.range(tf.cast(HEIGHT, dtype=tf.float32)):
        for j in tf.range(tf.cast(WIDTH, dtype=tf.float32)):

            # The pixel is obtained using the camera parameters
            PIXEL = EYE + zoom * FRONT + (-width / 2 + (j + 1) * pixel_length) * RIGHT + (
                        height / 2 - (i + 1) * pixel_length) * UP

            # The direction vector pointing from the EYE to the current PIXEL
            DIRECTION = PIXEL - EYE
            DIRECTION = DIRECTION / tf.norm(DIRECTION)

            # Some convenient quantities/renamings
            d = DIRECTION
            e = EYE
            Ad = A @ d
            Ae = A @ e
            res = Ae - b

            # These coefficients come about when attempting to find intersections of the ray with the surface
            a2 = (1 / 2) * tf.transpose(d) @ Ad
            a1 = tf.transpose(d) @ res
            a0 = (1 / 2) * tf.transpose(e) @ res - 1
            discriminant = a1 ** 2 - 4 * a2 * a0

            # We must use tf.cond instead of tf.where in order to avoid nan in the case of no intersection
            t_plus, t_minus = tf.cond(tf.greater(discriminant, 0.),
                                      lambda: ((-a1 + tf.sqrt(discriminant)) / (2 * a2),
                                               (-a1 - tf.sqrt(discriminant)) / (2 * a2)),
                                      lambda: (tf.zeros_like(a1), tf.zeros_like(a1)))

            # We are interested in the first point of intersection; these conditions are used to find it
            condition_0 = tf.logical_or(tf.less_equal(discriminant, 0.),
                                        tf.less_equal(tf.reduce_max([t_plus, t_minus]), 0.))
            condition_1 = tf.greater(tf.reduce_min([t_plus, t_minus]), 0.)
            condition_2 = tf.logical_and(tf.greater(t_plus, 0.), tf.greater(0., t_minus))
            condition_3 = tf.logical_and(tf.greater(t_minus, 0.), tf.greater(0., t_plus))

            value_0 = tf.constant(0., dtype=tf.float32)
            value_1 = tf.reduce_min([t_plus, t_minus])
            value_2 = t_plus
            value_3 = t_minus

            # Finally, the value of t such that EYE + t * DIRECTION is the first point of intersection with the surface
            t = tf.where(condition_0, value_0,
                         tf.where(condition_1, value_1,
                                  tf.where(condition_2, value_2,
                                           tf.where(condition_3, value_3, value_0))))

            # First point of intersection with surface
            POINT = EYE + t * DIRECTION

            # Direction from POINT to EYE
            V = -DIRECTION

            # Outward normal on surface at POINT
            N = A @ POINT - b
            N = N / tf.norm(N)
            N = tf.where(tf.greater_equal(tf.transpose(V) @ N, 0.), N, -N)

            # Direction vector from POINT to single light source
            L = LIGHT - POINT
            L = L / tf.norm(L)

            # Reflection of L about N
            R = 2 * (tf.transpose(L) @ N) * N - L

            # Intensity as computed from Phong reflectance model
            intensity = tf.where(condition_0, i_a,
                                 k_a * i_a + k_d * i_d * tf.transpose(L) @ N + k_s * i_s * (
                                             tf.transpose(R) @ V) ** shininess)
            intensity = tf.squeeze(intensity)

            # Update entry of image
            image = tf.tensor_scatter_nd_update(image, [[tf.cast(i, dtype=tf.int32), tf.cast(j, dtype=tf.int32)]],
                                                [intensity])

    return image

# Create a "true" surface to be reconstructed
a00 = tf.constant(1., dtype=tf.float32)
a01 = tf.constant(0., dtype=tf.float32)
a02 = tf.constant(0., dtype=tf.float32)
a11 = tf.constant(1., dtype=tf.float32)
a12 = tf.constant(0., dtype=tf.float32)
a22 = tf.constant(1., dtype=tf.float32)
b0 = tf.constant(0., dtype=tf.float32)
b1 = tf.constant(0., dtype=tf.float32)
b2 = tf.constant(0., dtype=tf.float32)
image = tf.Variable(tf.zeros(shape=(HEIGHT, WIDTH), dtype=tf.float32))
exact_image = get_image(a00, a01, a02, a11, a12, a22, b0, b1, b2, image)  # get_image makes use of image

# Initialize variables which will be modified via gradient descent
epsilon = 1e-1
a00 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=1.-epsilon, maxval=1.+epsilon)), dtype=tf.float32)
a01 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)
a02 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)
a11 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=1.-epsilon, maxval=1.+epsilon)), dtype=tf.float32)
a12 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)
a22 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=1.-epsilon, maxval=1.+epsilon)), dtype=tf.float32)
b0 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)
b1 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)
b2 = tf.Variable(tf.squeeze(tf.random.uniform(shape=(1,), minval=0.-epsilon, maxval=0.+epsilon)), dtype=tf.float32)

# Update surface variables using camera measurements
optimizer = Adam(learning_rate=1e-3)
epochs = 10
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        image = tf.Variable(tf.zeros(shape=(HEIGHT, WIDTH), dtype=tf.float32))
        recovered_image = get_image(a00, a01, a02, a11, a12, a22, b0, b1, b2, image)
        loss = tf.reduce_sum(tf.square(recovered_image - exact_image))
    grads = tape.gradient(loss, [a00, a01, a02, a11, a12, a22, b0, b1, b2])
    optimizer.apply_gradients(zip(grads, [a00, a01, a02, a11, a12, a22, b0, b1, b2]))
    print(f'Epoch: {epoch + 1}, Loss: {loss}')
