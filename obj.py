from classes import *
import numpy as np


class Face:
    __slots__ = 'vertex', 'normal'

    def __init__(self):
        self.vertex = list()
        self.normal = np.array([0, 0, 0])


class Object:
    __slots__ = 'filename', 'faces', 'mat', 'offset'

    def __init__(self, f='', m=Material(), o=[0, 0, 0]):
        self.filename = f
        self.faces = list()
        self.mat = m
        self.offset = np.array(o)

    def load_faces(self):
        # open obj file, load faces data
        vertex = list()
        normals = list()

        try:
            file = open(self.filename, 'r')
        except IOError:
            print("Could not read file : ", self.filename)
            return None

        for line in file:
            l = line.rstrip('\n')
            l = l.split(" ")  # split into ['x', 'float/vertex', 'float/vertex', 'float/vertex']
            l = [i for i in l if i]
            if len(l) == 0:
                continue

            if l[0] == "v":
                vertex.append([float(l[1]), float(l[2]), float(l[3])])

            if l[0] == "vn":
                normals.append([float(l[1]), float(l[2]), float(l[3])])

            if l[0] == "f":
                lp = l[1:]
                face = Face()
                n = lp[0].split('/')[2]  # split into []
                normal_check = True
                for s in lp:
                    s = s.split('/')
                    face.vertex.append(vertex[int(s[0])-1])

                    if n != s[2]:
                        normal_check = False

                if normal_check is True:
                    face.normal = normals[int(n)-1]

                self.faces.append(face)

        return None

    def triangulate(self):
        # turn polygon faces into triangles, ready to be rendered

        ret = np.array([])

        for f in self.faces:
            normal = [0, 0, 0]
            if not np.all(f.normal != np.array([0, 0, 0])):

                normal = f.normal

            if len(f.vertex) == 3:
                ret = np.append(ret, Triangle(f.vertex[0]+self.offset, f.vertex[1]+self.offset, f.vertex[2]+self.offset, normal, self.mat).compute_normal())

            if len(f.vertex) == 4:
                ret = np.append(ret, Triangle(f.vertex[0]+self.offset, f.vertex[1]+self.offset, f.vertex[2]+self.offset, normal, self.mat).compute_normal())
                ret = np.append(ret, Triangle(f.vertex[0]+self.offset, f.vertex[2]+self.offset, f.vertex[3]+self.offset, normal, self.mat).compute_normal())

        return ret
