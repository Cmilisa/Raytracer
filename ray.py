import numpy as np
import cv2

from render import *
from obj import *


nbr_pixels_y, nbr_pixels_x = 1024, 720  # size of the final image ;
fov = radians(45)  # real fov / 2
# int32 to avoid uint8 overflow when computing specular light
img = np.zeros((nbr_pixels_x, nbr_pixels_y, 3), dtype=np.int32)  # initializing the image numpy array

mat_dict = dict()  # material dictionary
# opencv uses a BGR format and not the standard RGB
mat_dict["Test"] = Material('Test', [255, 128, 128], 0.5, 0.25, 3.0)
mat_dict["Red"] = Material('Red', [162, 105, 255], 0.75, 0.1, 3.0)


light_list = list()  # all lights in the scene
light_list.append(Light([0, 0, 0], 1.0, 1.0))

# numpy array to contain all 3D geometry
scene_objects = np.array([])

# a few objects to render
# load_faces() method reads the obj file and lists all the mesh's faces, vertex and normals
# triangulate method translates above information into Triangle objects, outputs a Triangle numpy array

#scene_objects = np.append(scene_objects, ObjFile("monkey.obj", mat_dict["Test"], [0, 0, -5]).load_faces().triangulate())
scene_objects = np.append(scene_objects, ObjFile("cube.obj", mat_dict["Red"], [-1, -2, -5]).load_faces().triangulate())
#scene_objects = np.append(scene_objects, ObjFile("Love.obj", mat_dict["Red"], [0, -40, -100]).load_faces().triangulate())

#scene_objects.append(Sphere([0, 0, -9], 1.5, mat_dict["Red"]))
#scene_objects.append(Sphere([0, -5, -9], 2.0, mat_dict["Red"]))
#scene_objects.append(Sphere([0, 5, -9], 0.75, mat_dict["Test"]))
#scene_objects.append(Triangle([1, -1, -7], [0, 4, -7], [0, -2, -7], [0, 0, 0], mat_dict["Test"]).compute_normal())


def color_cut(a):
    """Takes in a numpy array of any shape, outputs a numpy array of the same shape
    Each element of the array is checked to be greater than 255 or lesser than 0, and if one condition is not met,
    makes it equal to 0 or 255. This function is needed to avoid misinterpretation of the color data when converting
    from int32 to uint8"""
    temp = np.where(a > 255, 255, a)
    return np.where(temp < 0, 0, temp)


def render():
    """Main function of image rendering. It generates the rays then calls the rest of the program to cast them"""
    # generates a matrix of tuples, with the first element i being the ith pixel along the horizontal axis
    # and j being the jth pixel along the vertical axis
    i, j = np.indices((nbr_pixels_y, nbr_pixels_x))
    coord_matrix = np.array((j, i))
    coord_matrix = coord_matrix.T
    # we add 0.5 to get the center of the pixels
    coord_matrix = coord_matrix + 0.5
    # generating a matrix with each element being the direction vector of the ray for that pixel
    indices = np.zeros((nbr_pixels_x, nbr_pixels_y, 3))
    indices[:coord_matrix.shape[0], :coord_matrix.shape[1], :coord_matrix.shape[2]] = coord_matrix
    indices[:, :, 0] = ((indices[:, :, 0]*2/nbr_pixels_x) - 1)*tan(fov)*(nbr_pixels_x/nbr_pixels_y)
    indices[:, :, 1] = ((indices[:, :, 1]*2/nbr_pixels_y) - 1)*tan(fov)
    indices[:, :, 2] = -1  # looking in the z = -1 direction
    indices = normalize(indices)
    img = cast_ray(indices, np.array([0, 0, 0]), scene_objects, light_list)

    cv2.imshow("render", color_cut(img).astype(np.uint8))  # uint8 because opencv can't handle anything else
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    render()
