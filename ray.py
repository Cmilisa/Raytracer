import numpy as np
import cv2

from classes import *
from render import *
from obj import *


# global variables
nbr_pixels_y, nbr_pixels_x = 1024, 720
fov = radians(45) # real fov / 2
# img = np.zeros((nbr_pixels_x, nbr_pixels_y, 3))
# int32 to avoid uint8 overflow when computing specular light
img = np.zeros((nbr_pixels_x, nbr_pixels_y, 3), dtype=np.int32)


mat_list = dict()  # material dictionnary
# mat_list["Test"] = Material('Test', Vec3(1.0, 0.5, 0.5), 1.0, 0.25, 1.0) #BGR and not RGB !!
mat_list["Test"] = Material('Test', [255, 128, 128], 0.5, 0.25, 3.0)
mat_list["Red"] = Material('Red', [162, 105, 255], 0.25, 0.25, 3.0)


light_list = list() # all lights in the scene
light_list.append(Light([0, 0, 0], 1.0, 1.0))

scene_objects = list()

"""monkey = Object("monkey.obj", mat_list["Test"], [0, 0, -5])
monkey.load_faces()
monkey_triangles = monkey.triangulate()
scene_objects = np.ndarray.tolist(np.append(np.asarray(scene_objects), monkey_triangles))"""

"""cube = Object("cube.obj", mat_list["Red"], [-1, -2, -5])
cube.load_faces()
cube_triangles = cube.triangulate()
scene_objects = np.ndarray.tolist(np.append(np.asarray(scene_objects), cube_triangles))"""

heart = Object("Love.obj", mat_list["Red"], [0, -40, -100])
heart.load_faces()
heart_triangles = heart.triangulate()
scene_objects = np.ndarray.tolist(np.append(np.asarray(scene_objects), heart_triangles))

#scene_objects.append(Sphere([0, 0, -9], 1.5, mat_list["Red"]))
#scene_objects.append(Sphere([0, -5, -9], 2.0, mat_list["Red"]))
#scene_objects.append(Sphere([0, 5, -9], 0.75, mat_list["Test"]))
#scene_objects.append(Triangle([1, -1, -7], [0, 4, -7], [0, -2, -7], [0, 0, 0], mat_list["Test"]).compute_normal()) #triangles must be defined clockwise


def recenter(a):
    temp = np.where(a > 255, 255, a)
    return np.where(temp < 0, 0, temp)


def render():
    i, j = np.indices((nbr_pixels_y, nbr_pixels_x))
    zipped = np.array((j, i))
    zipped = zipped.T
    temp = zipped + 0.5
    indices = np.zeros((nbr_pixels_x, nbr_pixels_y, 3))
    indices[:temp.shape[0], :temp.shape[1], :temp.shape[2]] = temp
    indices[:, :, 0] = ((indices[:, :, 0]*2/nbr_pixels_x) - 1)*tan(fov)*(nbr_pixels_x/nbr_pixels_y)
    indices[:, :, 1] = ((indices[:, :, 1]*2/nbr_pixels_y) - 1)*tan(fov)
    indices[:, :, 2] = -1
    indices = normalize(indices)
    img = cast_ray(indices, np.array([0, 0, 0]), scene_objects, light_list)

    cv2.imshow("render", recenter(img).astype(np.uint8))  # uint8 because opencv can't handle anything else
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    render()
