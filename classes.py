from math import *
from render import *


class Material:
    __slots__ = 'name', 'color', 'idiffuse', 'ispecular', 'alpha'

    def __init__(self, n='', c=[0, 0, 0], d=0.0, s=0.0, alpha=1.0):
        self.name = n
        self.color = np.array(c)
        self.idiffuse = d  # diffuse intensity
        self.ispecular = s  # specular intensity
        self.alpha = alpha  # alpha coefficient


class Light:
    __slots__ = 'pos', 'diffuse_intensity', 'specular_intensity'

    def __init__(self, pos=[0, 0, 0], id=0.0, ispe=0.0):
        self.pos = np.array(pos)
        self.diffuse_intensity = id
        self.specular_intensity = ispe


class Sphere:
    __slots__ = 'center', 'radius', 'mat'

    def __init__(self, center=[0, 0, 0], radius=0.0, m=Material()):
        self.center = np.array(center)
        self.radius = radius
        self.mat = m

    def compute_ray(self, ray_dir, ray_origin):
        """Computes the points of intersection of a given ray with the Sphere object.
        Variables with suffix v are vectors, and variables with suffix d are distances"""
        v_origin_center = self.center - ray_origin  # vector from the ray origin to the sphere center
        # distance between the origin and the sphere's center projection along the ray
        dot_dir_center = np.dot(ray_dir, v_origin_center)
        # distance between the sphere's center and its projection on the ray
        d_center_proj = vec_norm(v_origin_center - mul_vec_scalar_matrices(ray_dir, dot_dir_center))

        # if the projection of the sphere center is farther than the radius, then obviously it is not in the sphere
        d_center_proj = np.where(d_center_proj > self.radius, np.nan, d_center_proj)
        # else compute the distance between the projection point and the intersection point of the ray
        d_proj_inter = np.sqrt(self.radius * self.radius - d_center_proj * d_center_proj)
        # distance between the origin and the intersection point (2 possible points, we take the first one)
        d_origin_inter = dot_dir_center - d_proj_inter

        # if the first point is behind the camera we take the second one
        d_origin_inter = np.where(d_origin_inter < 0, dot_dir_center + d_proj_inter, d_origin_inter)
        # if the second one is behind, then the whole sphere is behind
        d_origin_inter = np.where(d_origin_inter < 0, np.nan, d_origin_inter)

        # calculate the intersection point's coordinates
        temp = extend_matrix(d_origin_inter)
        ret = ray_origin + ray_dir * temp

        return ret

    def is_point_contained(self, point):
        """Checks if the input point is inside the sphere, thus being a possible intersection point. Is used in
        lighting checks"""
        return np.where(vec_norm(self.center - point) <= self.radius + 1, True, False)

    def compute_light(self, ray_dir, ray_origin, point, object_list, light_list):
        """Computes the color of a pixel for the object given intersection points"""
        # normal vector to the sphere in the point
        normal = normalize(point - self.center)
        v_point_light = normalize(light_list[0].pos - point)

        # initializing lighting variables
        diffuse_intensity = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))
        specular_intensity = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))

        for l in light_list:
            d_point_light = dot_product(v_point_light, normal)

            # scalar product used in specular light calculation
            dotted = dot_product(normalize((normal * extend_matrix(d_point_light) * 2) - v_point_light), ray_dir) * (-1)

            diffuse_intensity = np.where(d_point_light >= 0, self.mat.idiffuse*d_point_light*l.diffuse_intensity, 0)

            specular_intensity = np.where(d_point_light >= 0, specular_intensity, 0)
            specular_intensity = np.where(dotted >= 0, (dotted**self.mat.alpha)*l.specular_intensity*self.mat.ispecular, 0)

        final = extend_matrix(diffuse_intensity) * self.mat.color \
                + extend_matrix(specular_intensity) * np.array([255, 255, 255], dtype=np.int32)

        return np.where(extend_matrix(self.is_point_contained(point)), final, np.array([0, 0, 0]))


class Triangle:
    __slots__ = 'p1', 'p2', 'p3', 'normal', 'mat'

    def __init__(self, a=[0, 0, 0], b=[0, 0, 0], c=[0, 0, 0], n=[0, 0, 0], m=Material()):
        self.p1 = np.array(a)
        self.p2 = np.array(b)
        self.p3 = np.array(c)
        self.normal = np.array(n)
        self.mat = m

    def compute_normal(self):
        """Computes the normal of the triangle is needed. Assumes the points are defined clockwise"""
        v1 = self.p2 - self.p1
        v2 = self.p3 - self.p1
        self.normal = np_normalize(np.cross(v1, v2))

        return self

    def is_point_contained(self, point):
        """Checks if the input point is inside the triangle, thus being a possible intersection point. Is used in
        lighting checks"""
        cross1 = dot_matrix_vec(cross(self.p2 - self.p1, point - self.p1), self.normal)
        cross2 = dot_matrix_vec(cross(self.p3 - self.p2, point - self.p2), self.normal)
        cross3 = dot_matrix_vec(cross(self.p1 - self.p3, point - self.p3), self.normal)
        final = np.where(cross1 <= 0, 0, 1)
        final = final + np.where(cross2 <= 0, 0, 1)
        final = final + np.where(cross3 <= 0, 0, 1)
        final = np.where(final >= 3, True, False)

        return final

    def compute_ray(self, ray_dir, ray_origin):
        """Computes the point of intersection of the ray with the triangle"""
        # First check if the ray intersection with the triangle plane
        # Check if the ray is parallel to the plane
        dirn = dot_matrix_vec(ray_dir, self.normal)
        ret = np.where(dirn == 0, np.nan, dirn)

        # Computes the distance along the ray of the intersection point with the plane
        if len(ray_origin.shape) is 1:
            d_ray = (np.dot(self.normal, ray_origin) + np.dot(self.normal, self.p1)) / ret
        else:
            d_ray = (dot_matrix_vec(ray_origin, self.normal) + np.dot(self.normal, self.p1)) / ret

        # is the point behind the camera ?
        ret = np.where(d_ray < 0, np.nan, d_ray)
        # coordinates of the point of intersection
        point = ray_origin + mul_vec_scalar_matrices(ray_dir, ret)

        # is point inside triangle ?
        final = extend_matrix(self.is_point_contained(point))
        final[:, :, 0] = np.where(final[:, :, 0] == True, point[:, :, 0], np.nan)
        final[:, :, 1] = np.where(final[:, :, 1] == True, point[:, :, 1], np.nan)
        final[:, :, 2] = np.where(final[:, :, 2] == True, point[:, :, 2], np.nan)

        return final

    def compute_light(self, ray_dir, ray_origin, point, object_list, light_list):
        """Computes the color of a pixel for the object given intersection points"""
        # initializing lighting variables
        diffuse_intensity = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))
        specular_intensity = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))

        for l in light_list:
            # this is the same method as the sphere compute_light method
            v_point_light = normalize(l.pos - point)
            d_point_light = dot_matrix_vec(v_point_light, self.normal)

            dotted = dot_product(normalize((self.normal * extend_matrix(d_point_light) * 2) - v_point_light), ray_dir) * (-1)

            diffuse_intensity = np.where(d_point_light >= 0, self.mat.idiffuse * d_point_light * l.diffuse_intensity, 0)

            specular_intensity = np.where(d_point_light >= 0, specular_intensity, 0)
            specular_intensity = np.where(dotted >= 0, (dotted ** self.mat.alpha) * l.specular_intensity * self.mat.ispecular, 0)

        final = extend_matrix(diffuse_intensity) * self.mat.color \
                + extend_matrix(specular_intensity) * np.array([255, 255, 255], dtype=np.int32)

        return np.where(extend_matrix(self.is_point_contained(point)), final, np.array([0, 0, 0]))
