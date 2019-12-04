import math


def rotate_wrt_origin(xy, theta):
    return (xy[0] * math.cos(theta) + xy[1] * math.sin(theta),
            -xy[0] * math.sin(theta) + xy[1] * math.cos(theta))