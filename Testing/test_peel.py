import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NeuroPET"))
from NeuroPETLib import peel  # noqa: E402


def _sphere(flipX=True):
    n = 41
    m = np.diag([-1.0 if flipX else 1.0, 1.0, 1.0, 1.0])
    m[:3, 3] = [20.0 if flipX else -20.0, -20.0, -20.0]       # RAS 0 at the centre voxel
    k, j, i = np.indices((n, n, n))
    ras = peel.rasOf(np.stack([k, j, i], axis=-1), m)
    mask = np.linalg.norm(ras, axis=-1) <= 15.0
    return mask, m, ras


def test_axes_and_depth():
    mask, m, _ = _sphere()
    info = peel.axes(m)
    assert info[0] == (2, -1, 1.0) and info[2] == (0, 1, 1.0)
    depth = peel.depthMap(mask, m)
    assert depth.max() > 14 and depth[0, 0, 0] == 0


def test_peel_top_and_sides():
    mask, m, ras = _sphere()
    depth = peel.depthMap(mask, m)
    for key, axis, sign in (("Top", 2, 1), ("Left", 0, -1), ("Right", 0, 1), ("Ant", 1, 1), ("Post", 1, -1)):
        kji, hit, spacing = peel.peel(depth, 0.0, m, key)
        assert spacing == (1.0, 1.0)
        centre = tuple(s // 2 for s in hit.shape)
        assert hit[centre] and not hit[0, 0]
        point = ras[tuple(kji[centre])]
        assert abs(point[axis] * sign - 15.0) < 1.01            # the surface on the camera's side
        deeper, _, _ = peel.peel(depth, 5.0, m, key)
        assert abs(ras[tuple(deeper[centre])][axis] * sign - 10.0) < 1.51
    # the superior view: front up, the patient's right on the right
    kji, hit, _ = peel.peel(depth, 0.0, m, "Top")
    rows, cols = hit.shape
    assert ras[tuple(kji[rows // 2, cols - 8])][0] > 0          # right column: patient right (+x)
    assert ras[tuple(kji[8, cols // 2])][1] > 0                 # top row: anterior (+y)
    # the left view: front on the left of the screen
    kji, hit, _ = peel.peel(depth, 0.0, m, "Left")
    assert ras[tuple(kji[hit.shape[0] // 2, 8])][1] > 0


def test_take_sample_mask():
    mask, m, ras = _sphere(flipX=False)
    labels = np.zeros(mask.shape, np.int16)
    labels[mask] = 1
    labels[20, 20, 20] = 0          # a hole: filled
    assert peel.brainMask(labels)[20, 20, 20]
    depth = peel.depthMap(peel.brainMask(labels), m)
    kji, hit, _ = peel.peel(depth, 2.0, m, "Right")
    values = peel.take(ras[..., 0].astype(np.float32), kji, hit, fill=-1)
    assert (values[hit] > 0).all() and (values[~hit] == -1).all()
    sampled = peel.sample(ras[..., 0], m, peel.rasOf(kji, m))
    assert np.allclose(sampled[hit], values[hit])


def test_unfold_sphere():
    mask, m, ras = _sphere()
    depth = peel.depthMap(mask, m)
    rays = peel.unfoldRays(depth, m, "Ant", width=64)
    kji, hit, (rowMm, colMm) = peel.unfold(rays, 0.0, m)
    assert hit.shape == (32, 64) and rowMm == colMm and 0.5 < colMm < 1.5
    assert hit[16, 32] and not hit[0, 0]
    radii = np.linalg.norm(ras[tuple(kji[hit].T)], axis=-1)
    assert np.all(np.abs(radii - 15.0) < 1.6)                   # the whole map on the surface
    deeper, hitDeep, _ = peel.unfold(rays, 5.0, m)
    assert np.all(np.abs(np.linalg.norm(ras[tuple(deeper[hitDeep].T)], axis=-1) - 10.0) < 1.6)
    # the centre of the map the front, the top superior, the right of the map the patient's left (seen from outside)
    assert ras[tuple(kji[16, 32])][1] > 13
    assert ras[tuple(kji[1, 32])][2] > 13
    assert ras[tuple(kji[16, 48])][0] < -10
    left = peel.unfold(peel.unfoldRays(depth, m, "Left", width=64), 0.0, m)[0]
    assert ras[tuple(left[16, 32])][0] < -13 and ras[tuple(left[16, 48])][1] < -10     # left: its right the back


def test_unfold_scroll():
    mask, m, ras = _sphere()
    depth = peel.depthMap(mask, m)
    rays = peel.unfoldRays(depth, m, "Scroll", width=40)
    kji, hit, (rowMm, colMm) = peel.unfold(rays, 0.0, m)
    rows, cols = hit.shape
    assert cols == 40 and rows == round((255 + 30) / 180 * 40)
    assert hit[:, cols // 2].all() and not hit[0, 0] and not hit[-1, -1]       # the sides rounded
    point = lambda r, c: ras[tuple(kji[r, c])]      # noqa: E731
    assert point(int(30 / 285 * rows), cols // 2)[1] > 13          # frontal pole near the top
    assert point(int(120 / 285 * rows), cols // 2)[2] > 13         # the vertex
    assert point(rows - 1, cols // 2)[2] < -13                     # cerebellum (down) at the bottom
    assert point(int(120 / 285 * rows), 0)[0] < -13                 # left on the left
    assert point(int(120 / 285 * rows), cols - 1)[0] > 13


def test_sample_linear():
    m = np.eye(4)
    a = np.zeros((3, 3, 3), np.float32)
    a[1, 1, 2] = 10.0
    assert abs(peel.sample(a, m, np.array([[1.5, 1.0, 1.0]]), order=1)[0] - 5.0) < 1e-5     # halfway in i
    assert peel.sample(a, m, np.array([[1.4, 1.0, 1.0]]))[0] == 0.0                        # nearest
    rgb = np.zeros((3, 3, 3, 3), np.uint8)
    rgb[1, 1, 2] = (200, 100, 0)
    assert tuple(peel.sample(rgb, m, np.array([[1.5, 1.0, 1.0]]), order=1)[0]) == (100, 50, 0)


def test_unfold_box():
    mask, m, ras = _sphere()
    depth = peel.depthMap(mask, m)
    directions, inside, _ = peel.boxDirections(degreesPerPixel=3.0)
    rows, cols = inside.shape
    assert cols == round((90 + 2 * 105) / 3) and rows == round((90 + 105 + 120) / 3)
    assert not inside[0, 0] and not inside[-1, -1] and inside[rows // 2 - 5, cols // 2]
    rays = peel.unfoldRays(depth, m, "Box")
    kji, hit, _ = peel.unfold(rays, 0.0, m)
    r, c = hit.shape
    top = round(105 / 0.6)                       # first row of the superior face
    centre = (top + round(45 / 0.6), c // 2)
    point = lambda rc: ras[tuple(kji[rc])]      # noqa: E731
    assert hit[centre] and point(centre)[2] > 13                             # superior in the middle
    assert point((top - round(45 / 0.6), c // 2))[1] > 13                    # front above (horizontal front)
    assert point((r - 1, c // 2))[1] < -2 and point((r - 1, c // 2))[2] < -13   # the back, low (cerebellum) at the bottom
    assert point((centre[0], c - 1))[0] > 2 and point((centre[0], c - 1))[2] < -12    # right side on the right
    assert point((centre[0], 0))[0] < -2
    # continuous over the fold between the superior face and the front
    assert np.linalg.norm(point((top, c // 2)) - point((top - 1, c // 2))) < 2.0


def test_unfold_mercator():
    directions, inside, perPixel = peel.mercatorDirections(width=40)
    rows, cols = inside.shape
    assert inside.all() and rows == round((255 + 30) / 180 * 40)
    assert abs(cols * perPixel / 2 - np.log(np.tan(np.radians(45 + 40)))) < 2 * perPixel
    assert directions[rows // 2, 0, 0] < -0.95 and directions[rows // 2, -1, 0] > 0.95     # the ears at the sides
    assert directions[0, cols // 2, 1] > 0.8                                               # frontal on the top


def test_unfold_poles_mercator():
    directions, inside, perPixel = peel.polesMercatorDirections(degreesPerPixel=3.0)
    rows, cols = inside.shape
    assert cols == 110 and inside.all() and "PolesMercator" in peel.WIDE_MAPS
    assert directions[0, cols // 2, 1] > 0.98 and directions[-1, cols // 2, 1] < -0.98    # frontal top, occipital bottom
    assert directions[rows // 2, cols // 2, 2] > 0.99                                      # superior in the middle
    assert directions[rows // 2, cols // 2 + round(90 / 3), 0] > 0.99                      # the right on the right
    assert abs(np.degrees(np.arcsin(directions[0, 0, 1])) - 80.0) < 2.0                    # cut at 80 degrees


def test_unfold_poles():
    directions, inside, perPixel = peel.polesDirections(degreesPerPixel=3.0)
    rows, cols = inside.shape
    assert rows == 60 and cols == 110 and "Poles" in peel.WIDE_MAPS
    assert peel.UNFOLD_CENTERS[0][0] == "Poles"                                            # the default
    assert inside[rows // 2].all() and not inside[0, 0] and inside[0, cols // 2]          # rounded corners
    assert directions[0, cols // 2, 1] > 0.99 and directions[-1, cols // 2, 1] < -0.99    # frontal top, occipital bottom
    assert directions[rows // 2, cols // 2, 2] > 0.99                                      # superior in the middle
    right = directions[rows // 2, cols // 2 + round(90 / 3)]
    assert right[0] > 0.99                                                                 # the right on the right
