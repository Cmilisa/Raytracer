import numpy as np


def cast_ray(ray_dir, ray_origin, object_list, light_list):
    """Function that takes in rays information (origin and direction) and 3D geometry information, and computes
    each ray's intersection with the scene and the pixel's color. Outputs the final image to be displayed"""

    # initializing the matrix that contains the coordinates of the nearest points of intersection for a give pixel
    nearest_points = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))
    nearest_points.fill(10000)
    # stores the distance to the origin of the nearest point
    norm_min = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))
    norm_min.fill(10000.)

    for i in range(len(object_list)):  # loop through all objects and compute the rays intersection with each one
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        computed_norm = vec_norm(computed)

        # extend_matrix is a function that broadcasts (x,y) shape matrices into (x,y,3) matrices
        computed_norm_extended = extend_matrix(computed_norm)
        norm_min_extended = extend_matrix(norm_min)

        nearest_points = np.where(computed_norm_extended < norm_min_extended, computed, nearest_points)
        norm_min = np.where(computed_norm < norm_min, computed_norm, norm_min)

    # rendered image
    render = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))

    for i in range(len(object_list)):
        # loop through all objects and compute lighting for each one given a point of intersection
        render = render + object_list[i].compute_light(ray_dir, ray_origin, nearest_points, object_list, light_list)

    # if the pixel is black for all objects (meaning there was no intersection), replace it with background color
    render = np.where(render == np.array([0, 0, 0]), np.array([150, 250, 125]), render)

    return render


# Utility functions

def normalize(vec):
    """Takes in a matrix where each element is a vector. Outputs a matrix of the same shape, with each vector
    normalized"""
    return div_vec_scalar_matrices(vec, np.sqrt(dot_product(vec, vec)))


def np_normalize(vec):
    """Does the same as normalize(), but using the numpy dot function instead. Used for simple (3) shape vectors
    normalization"""
    return vec / np.sqrt(np.dot(vec, vec))


def div_vec_scalar_matrices(vec, a):
    """Takes in a matrix with vector elements, and divides each vector component by a scalar. For some reason,
    just using vec / a didn't work"""
    temp = np.copy(vec)
    temp[:, :, 0] /= a
    temp[:, :, 1] /= a
    temp[:, :, 2] /= a
    return temp


def mul_vec_scalar_matrices(v1, v2):
    """Is the same as div_vec_scalar_matrices but for multiplication"""
    temp = np.copy(v1)
    temp[:, :, 0] *= v2
    temp[:, :, 1] *= v2
    temp[:, :, 2] *= v2
    return temp


def vec_norm(vec):
    """Takes in a matrix of shape (x, y, 3), and outputs a matrix of shape (x, y) with the elements being the norm
    of the original matrix vectors"""
    return np.sqrt(dot_product(vec, vec))


def dot_product(v1, v2):
    """Custom dot product function for vector matrices"""
    return v1[:, :, 0] * v2[:, :, 0] + v1[:, :, 1] * v2[:, :, 1] + v1[:, :, 2] * v2[:, :, 2]


def extend_matrix(m):
    """Broadcasts a (x, y) shape matrix into a (x, y, 3) shape matrix"""
    temp = np.zeros((m.shape[0], m.shape[1], 3))
    temp[:, :, 0] = m
    temp[:, :, 1] = m
    temp[:, :, 2] = m
    return temp


def dot_matrix_vec(v1, v2):
    """Computes the dot product of each vector of a (x, y, 3) matrix with one vector v2"""
    return v1[:, :, 0] * v2[0] + v1[:, :, 1] * v2[1] + v1[:, :, 2] * v2[2]


def cross(v1, v2):
    """Computes the cross product of each vector of a (x, y, 3) matrix with one vector v2"""
    temp = np.copy(v2)
    temp[:, :, 0] = v1[1] * v2[:, :, 2] - v1[2] * v2[:, :, 1]
    temp[:, :, 1] = v1[2] * v2[:, :, 0] - v1[0] * v2[:, :, 2]
    temp[:, :, 2] = v1[0] * v2[:, :, 1] - v1[1] * v2[:, :, 0]
    return temp
