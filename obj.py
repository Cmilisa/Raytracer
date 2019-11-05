from classes import *
import numpy as np


class Face:
    __slots__ = 'vertex', 'normal'

    def __init__(self):
        self.vertex = list()
        self.normal = np.array([0, 0, 0])


class ObjFile:
    __slots__ = 'filename', 'faces', 'mat', 'offset'

    def __init__(self, f='', m=Material(), o=[0, 0, 0]):
        self.filename = f
        self.faces = list()
        self.mat = m
        self.offset = np.array(o)

    def load_faces(self):
        """Opens the (self.filename).obj file and loads the geometry data"""
        vertex = list()
        normals = list()

        try:
            file = open(self.filename, 'r')
        except IOError:
            print("Could not read file : ", self.filename)
            return self

        for line in file:
            l = line.rstrip('\n')  # remove the ending special characters
            l = l.split(" ")  # split into ['x', 'float/vertex', 'float/vertex', 'float/vertex']
            l = [i for i in l if i]  # remove empty string in the resulting string array
            if len(l) == 0:  # if the array is empty (meaning it was an empty line)
                continue

            if l[0] == "v":  # vertex data
                vertex.append([float(l[1]), float(l[2]), float(l[3])])

            if l[0] == "vn":  # normal data
                normals.append([float(l[1]), float(l[2]), float(l[3])])

            if l[0] == "f":  # face data
                lp = l[1:]  # l[0] is "f" so we only take what's left
                face = Face()
                normal = lp[0].split('/')[2]  # select the normal vector index
                # temporary variable to check if normals at each point of the face are all the same, just in case
                normal_check = True
                for s in lp:  # store face vertex data into the Face object
                    s = s.split('/')
                    face.vertex.append(vertex[int(s[0])-1])

                    # if one normal was different
                    if normal != s[2]:
                        normal_check = False

                # if all the normals were the same
                if normal_check is True:
                    face.normal = normals[int(normal)-1]

                self.faces.append(face)

        return self

    def triangulate(self):
        """Translates face data contained in Face objects into Triangle objects to be rendered"""
        tri = np.array([])

        for f in self.faces:
            normal = [0, 0, 0]
            if not np.all(f.normal != np.array([0, 0, 0])):
                normal = f.normal

            # the face is already a triangle
            if len(f.vertex) == 3:
                tri = np.append(tri, Triangle(f.vertex[0]+self.offset, f.vertex[1]+self.offset, f.vertex[2]+self.offset, normal, self.mat).compute_normal())

            # if the space is a square. A proper triangulation algorithm is to be implemented
            if len(f.vertex) == 4:
                tri = np.append(tri, Triangle(f.vertex[0]+self.offset, f.vertex[1]+self.offset, f.vertex[2]+self.offset, normal, self.mat).compute_normal())
                tri = np.append(tri, Triangle(f.vertex[0]+self.offset, f.vertex[2]+self.offset, f.vertex[3]+self.offset, normal, self.mat).compute_normal())

        return ret
