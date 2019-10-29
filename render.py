from classes import *
import numpy as np


def cast_ray(ray_dir, ray_origin, object_list, light_list):
    object_list = np.array(object_list)
    """nearest_point = None
    index = 0
    temp = True
    for i in range(len(object_list)):
        point = object_list[i].compute_ray(ray_dir, ray_origin)

        if point is not None:
            if temp is True:
                nearest_point = point
                index = i
                temp = False
            if norm(point) < norm(nearest_point):
                nearest_point = point
                index = i"""

    """vcompute = np.vectorize(compute_ray)
    vnorm = np.vectorize(norm)
    object_array = np.array(object_list)
    test = vcompute(object_array, ray)
    test_p = np.where(test == None, Vec3(1000, 1000, 1000), test)
    predicate = vnorm(test_p)
    order = np.argsort(predicate)
    test_sorted = test[order]
    nearest_point = test_sorted[0]
    sorted_objects = object_array[order]"""

    """if nearest_point is not None:
        return object_list[index].compute_light(ray_dir, ray_origin, nearest_point, object_list, light_list)"""

    ret = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))
    ret.fill(np.nan)
    index = 0

    """for i in range(len(object_list)):
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        ret = computed"""

    nearest_points = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))
    nearest_points.fill(10000)
    nearest_index = np.zeros((ray_dir.shape[0], ray_dir.shape[1]), dtype=np.int32)
    nearest_index.fill(0)

    """for i in range(1, len(object_list)):
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        computed_norm = vec_norm(computed)
        computed_norm = np.where(np.isnan(computed_norm), 10000, computed_norm)

        nearest_norm = vec_norm(nearest_points)
        nearest_index = np.where(computed_norm < nearest_norm, i, nearest_index)

        computed_norm = extend_matrix(computed_norm)
        nearest_norm = extend_matrix(nearest_norm)
        nearest_points = np.where(computed_norm < nearest_norm, computed, nearest_points)"""

    final = np.zeros((ray_dir.shape[0], ray_dir.shape[1], 3))

    for i in range(len(object_list)):
        computed = object_list[i].compute_ray(ray_dir, ray_origin)
        final = final + object_list[i].compute_light(ray_dir, ray_origin, computed, object_list, light_list)


    """nearest_index_matrix = np.zeros((nearest_index.shape[0], nearest_index.shape[1], 3), dtype=np.int32)
    nearest_index_matrix[:, :, 0] = nearest_index
    nearest_index_matrix[:, :, 1] = nearest_index
    nearest_index_matrix[:, :, 2] = nearest_index
    test = np.zeros((nearest_index.shape[0], nearest_index.shape[1], 3), dtype=np.int32)


    for i in range(1, len(object_list)):
        test = np.where(nearest_index_matrix[:, :, 0] == i, object_list[i].compute_light(ray_dir, ray_origin, nearest_points, object_list, light_list), [255, 255, 255])"""

    """ret = np.where(nearest_points == 10000, np.nan, nearest_points)
    ret = np.where(np.isnan(ret) == False, object_matrix.compute_light(ray_dir, ray_origin, ret, object_list, light_list), np.array([150, 200, 125]))"""
    #ret = np.where(np.isnan(ret) == False, object_list[index].compute_light(ray_dir, ray_origin, ret, object_list, light_list), np.array([150, 200, 125]))
    #ret = np.where(np.isnan(ret) == False, object_list[nearest_index.astype(int)].compute_light(ray_dir, ray_origin, ret, object_list, light_list), np.array([150, 200, 125]))

    """if object_list.compute_ray(ray_dir, ray_origin) is not None:
        return object_list.mat.color"""

    final = np.where(final == np.array([0, 0, 0]), np.array([150, 250, 125]), final)

    return final  # background
    #return ret  # background


if __name__ == "__main__":
    o = list()
    l = list()
    m = Material()
    o.append(Background([200, 25, 125]))
    l.append(Light())
    ray_d = np.array([[[0, 0, -1], [0, 0, -1]], [[0, 0, -1], [0, 0, -1]]])
    t = cast_ray(ray_d, np.array([0, 0, 0]), o, l)
    print(t)
