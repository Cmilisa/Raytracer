from classes import *
import numpy as np


def cast_ray(ray_dir, ray_origin, object_list, light_list):
    object_list = np.array(object_list)

    nearest_points = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))
    nearest_points.fill(10000)
    nearest_index = np.zeros((ray_dir.shape[0], ray_dir.shape[1]), dtype=np.int32)
    nearest_index.fill(0)

    final = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))

    """for i in range(len(object_list)):
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        final = final + object_list[i].compute_light(ray_dir, ray_origin, computed, object_list, light_list)"""

    nearest_points = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))
    norm_min = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))
    norm_min.fill(10000.)

    for i in range(len(object_list)):
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        computed_norm = vec_norm(computed)

        computed_norm_extended = extend_matrix(computed_norm)
        norm_min_extended = extend_matrix(norm_min)

        nearest_points = np.where(computed_norm_extended < norm_min_extended, computed, nearest_points)
        norm_min = np.where(computed_norm < norm_min, computed_norm, norm_min)

    for i in range(len(object_list)):
        final = final + object_list[i].compute_light(ray_dir, ray_origin, nearest_points, object_list, light_list)

    final = np.where(final == np.array([0, 0, 0]), np.array([150, 250, 125]), final)

    return final  # background


if __name__ == "__main__":
    o = [Triangle([5, -5, -7], [-5, -5, -7], [0, -5, -7], [0, 0, 1], Material('Test', [255, 128, 128], 0.5, 0.25, 3.0)).compute_normal()]
    l = [Light()]
    ray_d = np.array([[[0., 0., -1.], [0., 0., -1.]], [[0., 0., -1.], [0., 0., -1.]]])
    t = cast_ray(ray_d, np.array([0., 0., 0.], dtype=np.float64), o, l)
    print(t)
