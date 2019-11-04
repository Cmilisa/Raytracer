# Python Raytracer

This is a simple Ray Tracing engine made with Python, Numpy and Opencv. It serves as an exercise and implementation of the mathematics of ray tracing, and of matrix parallel computation using numpy.

This engine can be used as a base for more advanced engines. It features Phong reflection model and basic materials settings.

# Mathematics of the engine

The goal of ray ray tracing is, for each pixel of the rendered image, to cast a virtual ray, determine the ray's point of intersection with the scene's objects, and compute the lightning with that information.
Two components define the rays that are casted :
1. A point of origin. Since translating the whole scene somewhere else in the world doesn't change the final rendered image, we'll take the origin's position (0, 0, 0), although this origin point is customizable.
2. A vector that represent the ray's direction. When taken from the ray's origin (which may not be (0, 0, 0)), this represent the tip of the arrow that would be the ray if we were to draw it. Thus, having the origin and the direction vector gives us an arrow with a know origin, magnitude (that should be normalized) and direction.

