import os
import timeit
import math

import numpy as np
import matplotlib.cm as cm

from numba import jit, guvectorize, float64
from PIL import Image


@jit('f8(f8, f8, f8)', cache=True)
def clamp(x, min_val, max_val):
    return min(max(min_val, x), max_val)


@jit('f8(f8[:], f8[:])', cache=True)
def dist(point1, point2):
    R = 6373.0

    lat1, lon1 = point1
    lat2, lon2 = point2

    delta = point2 - point1
    a = math.sin(delta[0] / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta[1] / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@guvectorize(['void(f8[:, :, :], f8[:], f8[:], f8, f8[:, :, :])'],
             '(a, b, c), (a), (b), () -> (a, b, c)',
             target='parallel', nopython=True)
def calc_grid(zeros, x_range, y_range, zoom, result):
    tile_count = 2.0 ** zoom
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            lon = x / tile_count * 2 * math.pi - math.pi
            lat = math.atan(math.sinh(math.pi * (1 - 2 * y / tile_count)))
            result[j][i] = (lat, lon)


@guvectorize(['void(f8[:, :, :], f8[:, :], f8[:, :])'],
             '(n, m, k), (a, b) -> (n, m)',
             target='parallel', nopython=True)
def calc_dists(positions, points, result):
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            buffer = 0
            for point in points:
                cur_dist = dist(positions[i][j], point)
                buffer += max(0, 1.2 - cur_dist)
            result[i][j] = min(1, buffer / 7)


@guvectorize(['void(f8[:, :, :], f8[:, :, :])'],
             '(n, m, k) -> (n, m, k)',
             target='parallel', nopython=True)
def calc_colors(dists, result):
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            x = dists[i][j][0]
            r = clamp((4.0 * x - 1.5) if x < 0.7 else (-4.0 * x + 4.5), 0, 1)
            g = clamp((4.0 * x - 0.5) if x < 0.5 else (-4.0 * x + 3.5), 0, 1)
            b = clamp((4.0 * x + 0.5) if x < 0.3 else (-4.0 * x + 2.5), 0, 1)
            a = clamp(dists[i][j][0] * 2, 0, 0.95)
            result[i][j] = (r, g, b, a)
    result *= 255


def gen_tile(zoom, x, y, points):
    size = 256
    rng = np.linspace(0, 1, size)

    grid = np.zeros((size, size, 2))
    calc_grid(grid, x + rng, y + rng, zoom, grid)

    points = np.radians(points)

    filtered_points = []
    for point in points:
        if (dist(point, grid[128][128]) - dist(grid[0][0], grid[128][128])) < 1.2:
            filtered_points.append(point)

    if len(filtered_points) == 0:
        return 'empty.png'

    dists = np.zeros((size, size))

    calc_dists(grid, filtered_points, dists)

    dists = np.dstack((
        dists.reshape((size, size, 1)),
        np.zeros((size, size, 3))
    ))

    colors = np.zeros_like(dists)
    calc_colors(dists, colors)

    image = Image.fromarray(colors.astype('uint8'))
    filename = 'files/{z}/{x}/{y}.png'.format(z=zoom, x=x, y=y)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    image.save(filename)
    return filename
