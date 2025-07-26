from __future__ import annotations

import cv2
import numpy as np
import torch
from matplotlib.lines import Line2D


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def draw_rectangle(ax, centre, theta, width, height, color=(1, 1, 1)):
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix(f'{c} {-s}; {s} {c}')

    p1 = [+width / 2, +height / 2]
    p2 = [-width / 2, +height / 2]
    p3 = [-width / 2, -height / 2]
    p4 = [+width / 2, -height / 2]
    p1_new = np.dot(p1, R) + centre
    p2_new = np.dot(p2, R) + centre
    p3_new = np.dot(p3, R) + centre
    p4_new = np.dot(p4, R) + centre

    rect_vertices = np.vstack([p1_new, p2_new, p3_new, p4_new, p1_new]).astype(
        np.float32,
    )
    line = Line2D(rect_vertices[:, 0], rect_vertices[:, 1], color=color)

    ax.add_line(line)


def compute_box_3d(obj, calib):
    """Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.yaw)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.x
    corners_3d[1, :] = corners_3d[1, :] + obj.y
    corners_3d[2, :] = corners_3d[2, :] + obj.z
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = calib.project_rect_to_image(np.transpose(corners_3d))
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    """Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

        i, j = k, k + 4
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return image
