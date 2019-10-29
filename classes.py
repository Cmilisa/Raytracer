from math import *
import numpy as np


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


def normalize(vec):
    return div_vec_scalar_matrices(vec, np.sqrt(dot_product(vec, vec)))


def div_vec_scalar_matrices(vec, a):
    temp = np.copy(vec)
    temp[:, :, 0] /= a
    temp[:, :, 1] /= a
    temp[:, :, 2] /= a
    return temp


def vec_norm(vec):
    return np.sqrt(dot_product(vec, vec))


def dot_product(v1, v2):
    return v1[:, :, 0] * v2[:, :, 0] + v1[:, :, 1] * v2[:, :, 1] + v1[:, :, 2] * v2[:, :, 2]


def mul_vec_scalar_matrices(v1, v2):
    temp = np.copy(v1)
    temp[:, :, 0] *= v2
    temp[:, :, 1] *= v2
    temp[:, :, 2] *= v2
    return temp


def extend_matrix(m):
    temp = np.zeros((m.shape[0], m.shape[1], 3))
    temp[:, :, 0] = m
    temp[:, :, 1] = m
    temp[:, :, 2] = m
    return temp


class Sphere:
    __slots__ = 'center', 'radius', 'mat'

    def __init__(self, center=[0, 0, 0], radius=0.0, m=Material()):
        self.center = np.array(center)
        self.radius = radius
        self.mat = m

    def compute_ray(self, ray_dir, ray_origin):
        AO = self.center - ray_origin  # vector from the ray origin to the sphere center
        dAC = np.dot(ray_dir, AO)
        # distance from the center of the sphere to its projection on the ray
        AC = mul_vec_scalar_matrices(ray_dir, dAC)
        CO = AO - AC
        dOC = vec_norm(CO)

        # if the projection of the sphere center is farther than the radius, then obviously it is not in the sphere
        dOC = np.where(dOC > self.radius, np.nan, dOC)
        dCP = np.sqrt(self.radius * self.radius - dOC * dOC)
        dAP = dAC - dCP

        dAP = np.where(dAP < 0, dAC + dCP, dAP)

        dAP = np.where(dAP < 0, np.nan, dAP)
        temp = extend_matrix(dAP)

        ret = ray_origin + ray_dir * temp
        return ret

    def is_point_contained(self, point):
        PC = self.center - point
        dPC = vec_norm(PC)
        return np.where(dPC <= self.radius + 1, True, False)

    def compute_light(self, ray_dir, ray_origin, point, object_list, light_list):
        diffuse_intensity = np.zeros((ray_dir.shape[0], ray_dir.shape[1]))
        specular_intensity = 0

        normal = normalize(point - self.center)
        p2l = normalize(light_list[0].pos - point)

        for l in light_list:
            temp = 1

            """occlu = self.center + (normal * 1.15)  # point a little outside the sphere to compensate float precision
            dir_occlu = normalize(l.pos - occlu)"""

            """for op in object_list:
                pointp = op.compute_ray(occlu, dir_occlu)

                if pointp is not None:
                    temp = 0"""

            p2ln = dot_product(p2l, normal)

            # reflected light ray from light source along the normal vector
            p2ln_temp = extend_matrix(p2ln)
            reflected = normalize((normal * p2ln_temp * 2) - p2l)

            # scalar product used in specular light calculation. The ray direction is multiplied by -1 because
            dotted = dot_product(reflected, ray_dir) * (-1)

            """if dot2(normal, p2l) >= 0:
                diffuse_intensity += self.mat.idiffuse*p2ln*l.diffuse_intensity  # diffuse intensity
                diffuse_intensity *= temp"""
            diffuse_intensity = np.where(p2ln >= 0, self.mat.idiffuse*p2ln*l.diffuse_intensity, 0)

            """if dotted >= 0:
                specular_intensity += l.specular_intensity*self.mat.ispecular*(dotted**self.mat.alpha)
                specular_intensity *= temp"""
            specular_intensity = np.where(p2ln >= 0, specular_intensity, 0)
            specular_intensity = np.where(dotted >= 0, (dotted**self.mat.alpha)*l.specular_intensity*self.mat.ispecular, 0)

        diffuse_temp = extend_matrix(diffuse_intensity)
        specular_temp = extend_matrix(specular_intensity)

        final = diffuse_temp * self.mat.color + specular_temp * np.array([255, 255, 255], dtype=np.int32)
        contained = self.is_point_contained(point)
        contained_extended = extend_matrix(contained)
        final = np.where(contained_extended, final, np.array([0, 0, 0]))

        return final


class Triangle:
    __slots__ = 'p1', 'p2', 'p3', 'normal', 'mat'

    def __init__(self, a=[0, 0, 0], b=[0, 0, 0], c=[0, 0, 0], n=[0, 0, 0], m=Material()):
        self.p1 = np.array(a)
        self.p2 = np.array(b)
        self.p3 = np.array(c)
        self.normal = np.array(n)
        self.mat = m

    def compute_normal(self):  # assuming points are defined clockwise
        v1 = self.p2 - self.p1
        v2 = self.p3 - self.p1
        self.normal = np.cross(v1, v2)
        normalize(self.normal)

        return self

    """def compute_ray(self, ray_dir, ray_origin):
        dirn = ray_dir * self.normal

        if (dirn) == 0:  # the ray and the triangle's plane are parrallel
            return None

        dR = (self.normal * ray_origin + self.normal * self.p1) / (dirn)  # distance of the plane along the ray direction

        if dR < 0:  # if the triangle is behind the ray's origin
            return None

        point = ray_origin + ray_dir * dR  # point of intersection with the triangle's plane

        if ((self.normal * cross(self.p2 - self.p1, point - self.p1)) > 0 and  # if the point of intersection is inside the triangle
            (self.normal * cross(self.p3 - self.p2, point - self.p2)) > 0 and
            (self.normal * cross(self.p1 - self.p3, point - self.p3)) > 0):
            return point

        return None"""

    def compute_ray(self, ray_dir, ray_origin):
        dirn = ray_dir * self.normal

        ret = np.where(dirn == 0, np.nan, dirn)

        dR = (self.normal * ray_origin + self.normal * self.p1) / ret

        ret = np.where(dR < 0, np.nan, ret)

        point = ray_origin + (ray_dir * ret)

        point = np.where(self.normal * np.cross(self.p2 - self.p1, point - self.p1) <= 0, np.nan, point)
        point = np.where(self.normal * np.cross(self.p3 - self.p2, point - self.p2) <= 0, np.nan, point)
        point = np.where(self.normal * np.cross(self.p1 - self.p3, point - self.p3) <= 0, np.nan, point)

        return point

    def compute_light(self, ray_dir, ray_origin, point, object_list, light_list):
        """diffuse_intensity = 0
        specular_intensity = 0

        for l in light_list:
            p2l = normalize(l.pos - point)
            p2lo = np.dot(p2l, self.normal)

            reflected = normalize((self.normal * 2 * p2lo) - p2l)
            dotted = np.dot(reflected, ray_dir) * (-1)

            if np.dot(self.normal, p2l) >= 0:
                diffuse_intensity += self.mat.idiffuse * p2lo * l.diffuse_intensity

                if dotted >= 0:
                    specular_intensity += l.specular_intensity * self.mat.ispecular * (dotted ** self.mat.alpha)

        return self.mat.color * diffuse_intensity + np.array([255, 255, 255], dtype=np.int32) * specular_intensity"""
        ret = np.where(np.isnan(point), np.array([125, 125, 125]), self.mat.color)
        return ret


class Background:
    __slots__ = 'color'

    def __init__(self, c=[150, 200, 125]):
        self.color = np.array(c)

    def compute_light(self, ray_dir, ray_origin, point, object_list, light_list):
        print("coucou")
        return self.color


if __name__ == "__main__":
    print("coucou")
