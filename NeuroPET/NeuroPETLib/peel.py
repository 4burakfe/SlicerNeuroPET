"""Curvilinear "peel" views of the Static Neuro Review (numpy + scipy, no Slicer).

The brain (the union of the regions, its holes filled) is peeled like the curvilinear reformatting of the MRI epilepsy
protocols: the shell depth mm under its surface (the voxels at least depth mm from the outside, Euclidean distance)
is seen flat from one side. For each pixel of a view (superior, anterior, posterior, left, right) a ray from that side
goes straight in and stops at the first voxel of the shell; the images (the anatomy, the PET, the scores, the region
labels) take the values there, so all of them show the same curved surface.

    grid        the review grid (k, j, i arrays) and its IJK to RAS matrix: axis-aligned (a diagonal or a permutation
                of one, flips allowed), as the MNI template grids
    peel(...)   per pixel (rows, columns): the voxel (k, j, i) of the grid hit and whether the ray hit the shell;
                rows from the top of the view down, columns from its left to its right (the cameras of ssp.CAMERAS:
                the superior view with the front up and the patient's right on the right, as the 3D-SSP)

The unfolded brain (one map of the whole surface, cerebellum included): rays from the brain's centre (the centroid of
the mask) in every direction. The scroll (the default, as the curved "scroll" reformat of the paediatric MRI flat-earth
map, Simpson et al. Pediatr Radiol 2016) is rectangular: the rows from the frontal pole on the top over the vertex to
the cerebellum on the bottom, the columns the angle from the midline (left hemisphere on the left). The world maps
are equal-area Hammer projections (latitude up: superior; the map's centre the side picked, seen from outside).
Each ray comes in from outside and stops where the depth first reaches depth mm, as the peel views.

    unfoldRays(depth, ijkToRas, center)   the depths along the rays of every pixel (made once a brain and centre)
                                          center: "Scroll" (the default), "Box" (the net of a cube, cut at its
                                          vertical corners) or a side for a world map
    unfold(rays, depthMm)                 per pixel the voxel (k, j, i) hit and whether the ray hit the shell
"""

import numpy as np

from . import ssp

# the maps (the first the default): (key, title, the centre side of a Hammer map or None)
UNFOLD_CENTERS = (("Poles", "Front-back poles (frontal top, occipital bottom, wide)", None),
                  ("Scroll", "Scroll (frontal top, cerebellum bottom)", None),
                  ("PolesMercator", "Front-back poles, Mercator (square corners, wide)", None),
                  ("Mercator", "Mercator (frontal top, cerebellum bottom)", None),
                  ("Box", "Box (top in the middle, sides folded out)", None),
                  ("Ant", "World map, front at the centre", (0.0, 1.0, 0.0)),
                  ("Post", "World map, back at the centre", (0.0, -1.0, 0.0)),
                  ("Left", "World map, left at the centre", (-1.0, 0.0, 0.0)),
                  ("Right", "World map, right at the centre", (1.0, 0.0, 0.0)))
UNFOLD_WIDTH = 512          # pixels of a world map (its height half that)
CLOSE_MM = 5.0             # the ball closing the regions into the brain's hull
BOX_DEG_PER_PIXEL = 0.6     # the box's pixels (degrees of a face)
BOX_LOW_DEG = -60.0         # the box's sides down to this elevation (degrees below the horizontal)
BOX_LOW_BACK_DEG = -75.0    # the back side further down: the cerebellum
POLES_DEG_PER_PIXEL = 0.5   # the front-back poles map's pixels
POLES_SEAM_DEG = 165.0      # its longitudes (around the front-back axis) up to +- this: the seam below, cut
WIDE_MAPS = ("Poles", "PolesMercator")   # the front-back poles maps: shown 2 x 2 in the layout
MERCATOR_LIMIT_DEG = 80.0   # the Mercator map's columns up to this angle from the midline
SCROLL_WIDTH = 320
SCROLL_ROUNDING = 3.0        # the scroll's outline: a superellipse of this power (a rounded rectangle)          # pixels of the scroll's width (its height from its arc)
# the scroll's rows: the angle in the sagittal plane from the front (0) over the vertex (90) and the back (180) down
# (degrees): from the orbitofrontal / frontal pole on the top row to the cerebellum on the bottom row
SCROLL_ARC_DEG = (-30.0, 255.0)
DEFAULT_DEPTH_MM = 13.0
MAX_DEPTH_MM = 30.0


def axes(ijkToRas):
    """{RAS axis (0 x, 1 y, 2 z): (numpy axis of a (k, j, i) array, +1 / -1: the RAS direction as that index grows,
    voxel size mm)} of an axis-aligned grid; ValueError for an oblique one."""
    m = np.asarray(ijkToRas, dtype=np.float64)[:3, :3]
    out = {}
    for t in range(3):          # i, j, k
        column = m[:, t]
        ras = int(np.argmax(np.abs(column)))
        size = float(abs(column[ras]))
        if size <= 0 or np.any(np.abs(np.delete(column, ras)) > 1e-3 * size) or ras in out:
            raise ValueError("The peel views need an axis-aligned grid.")
        out[ras] = (2 - t, 1 if column[ras] > 0 else -1, size)
    return out


def brainMask(labels, ijkToRas=None, closeMm=CLOSE_MM):
    """The brain of the peel: the regions (labels > 0) closed with a ball of closeMm (the sulci bridged: a smooth hull
    as the curvilinear reformatting's surface; the white matter out of the regions joined to them) and its holes
    filled (ventricles, white matter)."""
    from scipy import ndimage
    mask = np.asarray(labels) > 0
    if closeMm and closeMm > 0 and mask.any():
        sizes = [1.0, 1.0, 1.0]
        if ijkToRas is not None:
            for _, (axis, _, size) in axes(ijkToRas).items():
                sizes[axis] = size
        radii = [max(1, int(np.ceil(closeMm / size))) for size in sizes]
        grids = np.indices([2 * r + 1 for r in radii]).astype(np.float64)
        ball = sum(((grids[a] - radii[a]) * sizes[a]) ** 2 for a in range(3)) <= closeMm ** 2
        pad = [(r + 1, r + 1) for r in radii]
        closed = ndimage.binary_closing(np.pad(mask, pad), ball)
        mask = closed[tuple(slice(r + 1, -r - 1) for r in radii)] | mask
    return ndimage.binary_fill_holes(mask)


def depthMap(mask, ijkToRas):
    """Distance mm of every voxel of mask to the outside (float32; 0 outside)."""
    from scipy import ndimage
    info = axes(ijkToRas)
    sampling = [0.0, 0.0, 0.0]
    for _, (axis, _, size) in info.items():
        sampling[axis] = size
    return ndimage.distance_transform_edt(np.asarray(mask, bool), sampling=sampling).astype(np.float32)


def _direction(vector):
    """(RAS axis, sign) of an axis-aligned unit vector."""
    v = np.asarray(vector, dtype=np.float64)
    axis = int(np.argmax(np.abs(v)))
    return axis, (1 if v[axis] > 0 else -1)


def peel(depth, depthMm, ijkToRas, key):
    """(kji (rows, columns, 3) int, hit (rows, columns) bool, (row mm, column mm)) of the peel view key at depthMm:
    the voxel where each ray first reaches depth >= depthMm (the surface itself at 0)."""
    info = axes(ijkToRas)
    direction, up = ssp.CAMERAS[key]
    right = np.cross(-np.asarray(direction, np.float64), np.asarray(up, np.float64))
    order, flips, sizes = [], [], []
    for vector, rule in ((up, "up"), (right, "right"), (direction, "ray")):
        rasAxis, sign = _direction(vector)
        axis, grows, size = info[rasAxis]
        towards = grows * sign      # +1: the index grows along the vector
        # rows: the top (most up) first; columns: left to right (growing along right); the ray: from the camera in
        flip = towards > 0 if rule in ("up", "ray") else towards < 0
        order.append(axis)
        flips.append(flip)
        sizes.append(size)
    threshold = max(float(depthMm), 1e-3)
    inside = np.asarray(depth) >= threshold
    view = np.transpose(inside, order)
    for axis, flip in enumerate(flips):
        if flip:
            view = np.flip(view, axis=axis)
    first = np.argmax(view, axis=2)
    hit = view.any(axis=2)
    rows, cols = np.indices(first.shape)
    counts = view.shape
    position = [rows, cols, first]
    kji = np.zeros(first.shape + (3,), dtype=np.int64)
    for slot in range(3):
        index = position[slot]
        if flips[slot]:
            index = counts[slot] - 1 - index
        kji[..., order[slot]] = index
    return kji, hit, (sizes[0], sizes[1])


def rasOf(kji, ijkToRas):
    """RAS (..., 3) of voxel indices (k, j, i)."""
    kji = np.asarray(kji, dtype=np.float64)
    ijk = np.stack([kji[..., 2], kji[..., 1], kji[..., 0], np.ones(kji.shape[:-1])], axis=-1)
    return (ijk @ np.asarray(ijkToRas, dtype=np.float64).T)[..., :3]


def sample(array, ijkToRas, ras, fill=0, order=0):
    """The values of a (k, j, i[, c]) array at RAS points (order 0: the nearest voxel; 1: trilinear; fill outside the
    grid)."""
    array = np.asarray(array)
    inverse = np.linalg.inv(np.asarray(ijkToRas, dtype=np.float64))
    ras = np.asarray(ras, dtype=np.float64)
    ijk = ras @ inverse[:3, :3].T + inverse[:3, 3]
    if order:
        from scipy import ndimage
        coords = np.stack([ijk[..., 2].ravel(), ijk[..., 1].ravel(), ijk[..., 0].ravel()])
        channels = array.reshape(array.shape[:3] + (-1,))
        values = np.stack([ndimage.map_coordinates(channels[..., c].astype(np.float32, copy=False), coords,
                                                   order=1, mode="constant", cval=float(fill), prefilter=False)
                           for c in range(channels.shape[3])], axis=-1)
        values = values.reshape(ras.shape[:-1] + array.shape[3:])
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            values = np.clip(np.round(values), info.min, info.max)
        return values.astype(array.dtype)
    k, j, i = (np.floor(ijk[..., c] + 0.5).astype(np.int64) for c in (2, 1, 0))
    shape = array.shape[:3]
    valid = (k >= 0) & (k < shape[0]) & (j >= 0) & (j < shape[1]) & (i >= 0) & (i < shape[2])
    out = np.full(ras.shape[:-1] + array.shape[3:], fill, dtype=array.dtype)
    out[valid] = array[k[valid], j[valid], i[valid]]
    return out


def take(array, kji, hit, fill=0):
    """The values of a (k, j, i[, c]) array (on the review grid) at the peel's voxels (fill where no hit)."""
    array = np.asarray(array)
    out = np.full(hit.shape + array.shape[3:], fill, dtype=array.dtype)
    out[hit] = array[kji[hit, 0], kji[hit, 1], kji[hit, 2]]
    return out


def hammerInverse(width, height=None):
    """(longitude, latitude radians (height, width), inside bool) of the pixels of an equal-area Hammer map: rows from
    the top (north) down, columns from the left (west) to the right; outside the ellipse inside is False."""
    height = height or width // 2
    x = (np.arange(width) + 0.5) / width * 4.0 * np.sqrt(2.0) - 2.0 * np.sqrt(2.0)
    y = np.sqrt(2.0) - (np.arange(height) + 0.5) / height * 2.0 * np.sqrt(2.0)
    x, y = np.meshgrid(x, y)
    inside = (x / (2.0 * np.sqrt(2.0))) ** 2 + (y / np.sqrt(2.0)) ** 2 <= 1.0
    z = np.sqrt(np.clip(1.0 - (x / 4.0) ** 2 - (y / 2.0) ** 2, 0.0, None))
    lon = 2.0 * np.arctan2(z * x, 2.0 * (2.0 * z * z - 1.0))
    lat = np.arcsin(np.clip(z * y, -1.0, 1.0))
    return lon, lat, inside


def _centerVector(center):
    for key, _, vector in UNFOLD_CENTERS:
        if key == center:
            return None if vector is None else np.asarray(vector, np.float64)
    raise ValueError(f"Unknown centre of the unfolded map: {center}")


def scrollDirections(width=SCROLL_WIDTH, arc=SCROLL_ARC_DEG, rounding=SCROLL_ROUNDING):
    """(directions (rows, columns, 3) RAS unit vectors, inside (rows, columns) bool, radians a pixel) of the scroll:
    a map about the left-right axis. Rows: the angle in the sagittal plane from the frontal pole on the top over the
    vertex and the occiput to the cerebellum on the bottom; columns: the angle psi from the midline, the patient's left
    on the left (-90 degrees, straight left) to the right on the right (+90), as the superior view. A column's arc is
    drawn (1 - |psi / 90|^rounding) ** (1 / rounding) as high as the midline's: the map a rounded rectangle (a
    superellipse; 2: an ellipse; large: a rectangle), the lateral sides less stretched than on a rectangle."""
    perPixel = np.pi / width
    first, last = np.radians(arc[0]), np.radians(arc[1])
    middle = 0.5 * (first + last)
    height = int(round((last - first) / perPixel))
    offset = ((np.arange(height) + 0.5) - 0.5 * height) * perPixel        # from the middle row, radians of the midline
    psi = -0.5 * np.pi + (np.arange(width) + 0.5) * perPixel
    offset, psi = np.meshgrid(offset, psi, indexing="ij")
    scale = np.clip(1.0 - np.abs(psi / (0.5 * np.pi)) ** float(rounding), 0.0, None) ** (1.0 / float(rounding))
    theta = middle + offset / np.maximum(scale, 1e-6)
    inside = (theta >= first) & (theta <= last)
    sagittal = np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=-1)
    directions = np.cos(psi)[..., None] * sagittal
    directions[..., 0] = np.sin(psi)
    return directions, inside, perPixel


def polesDirections(degreesPerPixel=POLES_DEG_PER_PIXEL, seam=POLES_SEAM_DEG, rounding=SCROLL_ROUNDING):
    """(directions (rows, columns, 3), inside (rows, columns) bool, radians a pixel) of the front-back poles map: the
    poles at the frontal and occipital poles. Rows: the angle from the coronal plane, the frontal pole on the top to
    the occipital pole on the bottom; columns: the angle around the front-back axis, superior in the middle, the
    patient's right on the right (as the superior view), the inferior surface at the sides, the seam below cut at
    +- seam degrees. A row's width is (1 - |latitude / 90|^rounding) ** (1 / rounding) of the middle one's: a
    rounded rectangle, the poles not stretched across the map."""
    p = float(degreesPerPixel)
    rows = int(round(180.0 / p))
    columns = int(round(2.0 * float(seam) / p))
    latitude = np.radians(90.0 - (np.arange(rows) + 0.5) * p)
    x = np.radians(-float(seam) + (np.arange(columns) + 0.5) * p)      # longitude on the middle row
    latitude, x = np.meshgrid(latitude, x, indexing="ij")
    width = np.clip(1.0 - np.abs(latitude / (0.5 * np.pi)) ** float(rounding), 0.0, None) ** (1.0 / float(rounding))
    longitude = x / np.maximum(width, 1e-6)
    inside = np.abs(longitude) <= np.radians(float(seam))
    directions = np.stack([np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                           np.cos(latitude) * np.cos(longitude)], axis=-1)
    return directions, inside, np.radians(p)


def polesMercatorDirections(degreesPerPixel=POLES_DEG_PER_PIXEL, seam=POLES_SEAM_DEG, limit=MERCATOR_LIMIT_DEG):
    """(directions (rows, columns, 3), inside (all True), radians a pixel) of the front-back poles map as a Mercator
    map: columns as the poles map (the angle around the front-back axis, superior in the middle, the patient's right
    on the right, the seam below cut at +- seam degrees), rows the latitude through ln(tan(45 + latitude / 2)) from
    the frontal pole on the top to the occipital pole on the bottom, cut at +- limit degrees (the poles themselves
    infinitely far). Conformal, square corners: the shapes kept, near the poles enlarged 1 / cos(latitude) times."""
    perPixel = np.radians(float(degreesPerPixel))
    reach = np.log(np.tan(np.pi / 4.0 + np.radians(limit) / 2.0))
    rows = int(round(2.0 * reach / perPixel))
    columns = int(round(2.0 * np.radians(float(seam)) / perPixel))
    y = reach - (np.arange(rows) + 0.5) * perPixel
    latitude = 2.0 * np.arctan(np.exp(y)) - np.pi / 2.0
    longitude = np.radians(-float(seam)) + (np.arange(columns) + 0.5) * perPixel
    latitude, longitude = np.meshgrid(latitude, longitude, indexing="ij")
    directions = np.stack([np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                           np.cos(latitude) * np.cos(longitude)], axis=-1)
    return directions, np.ones(latitude.shape, bool), perPixel


def mercatorDirections(width=SCROLL_WIDTH, arc=SCROLL_ARC_DEG, limit=MERCATOR_LIMIT_DEG):
    """(directions (rows, columns, 3), inside (all True), radians a pixel) of the transverse Mercator map about the
    left-right axis (the poles at the ears): rows as the scroll (the sagittal angle, frontal pole on the top to the
    cerebellum on the bottom), columns the angle psi from the midline through ln(tan(45 + psi / 2)), up to +- limit
    degrees (the ears themselves infinitely far). Conformal: the shapes kept, the sides enlarged 1 / cos(psi) times
    both ways. width: pixels of the scroll's width (the same scale on the midline)."""
    perPixel = np.pi / width
    first, last = np.radians(arc[0]), np.radians(arc[1])
    height = int(round((last - first) / perPixel))
    reach = np.log(np.tan(np.pi / 4.0 + np.radians(limit) / 2.0))
    columns = int(round(2.0 * reach / perPixel))
    theta = first + (np.arange(height) + 0.5) * perPixel
    x = -reach + (np.arange(columns) + 0.5) * perPixel
    psi = 2.0 * np.arctan(np.exp(x)) - np.pi / 2.0
    theta, psi = np.meshgrid(theta, psi, indexing="ij")
    directions = np.stack([np.sin(psi), np.cos(psi) * np.cos(theta), np.cos(psi) * np.sin(theta)], axis=-1)
    return directions, np.ones(theta.shape, bool), perPixel


def boxDirections(degreesPerPixel=BOX_DEG_PER_PIXEL, low=BOX_LOW_DEG, lowBack=BOX_LOW_BACK_DEG):
    """(directions (rows, columns, 3) RAS unit vectors, inside (rows, columns) bool, radians a pixel) of the box: the
    brain unfolded as the net of a cube around it (equiangular cube faces, continuous across the folds), cut at its
    four vertical corners (right / left anterior, right / left posterior). In the middle the superior face (the front
    up, the patient's right on the right, as the superior view); folded out of it the sides down to low degrees below
    the horizontal (the posterior one to lowBack, for the cerebellum): the front (prefrontal, orbitofrontal) above,
    the right and left (lateral temporal) on the right and the left, the back (occipital, cerebellum) below."""
    flap = 45.0 - float(low)
    flapBack = 45.0 - float(lowBack)
    p = float(degreesPerPixel)
    width = int(round((90.0 + 2.0 * flap) / p))
    height = int(round((90.0 + flap + flapBack) / p))
    x = -45.0 - flap + (np.arange(width) + 0.5) * p
    y = 45.0 + flap - (np.arange(height) + 0.5) * p
    x, y = np.meshgrid(x, y)
    t = lambda degrees: np.tan(np.radians(degrees))      # noqa: E731
    vectors = np.zeros(x.shape + (3,))
    inside = np.zeros(x.shape, bool)
    middleX, middleY = np.abs(x) <= 45.0, np.abs(y) <= 45.0
    faces = (  # (pixels, (R, A, S) of the pixels' directions before normalization)
        (middleX & middleY, lambda: (t(x), t(y), np.ones_like(x))),                                   # superior
        (middleX & (y > 45.0) & (90.0 - y >= low), lambda: (t(x), np.ones_like(x), t(90.0 - y))),     # front
        (middleX & (y < -45.0) & (90.0 + y >= lowBack), lambda: (t(x), -np.ones_like(x), t(90.0 + y))),  # back
        (middleY & (x > 45.0) & (90.0 - x >= low), lambda: (np.ones_like(x), t(y), t(90.0 - x))),     # right
        (middleY & (x < -45.0) & (90.0 + x >= low), lambda: (-np.ones_like(x), t(y), t(90.0 + x))))   # left
    with np.errstate(over="ignore", invalid="ignore"):
        for pixels, vector in faces:
            r, a, sup = vector()
            vectors[pixels] = np.stack([r, a, sup], axis=-1)[pixels]
            inside |= pixels
    norms = np.linalg.norm(vectors, axis=-1)
    vectors[inside] /= norms[inside][:, None]
    return vectors, inside, np.radians(p)


class UnfoldRays:
    """The rays of an unfolded map: the depth (float16) at every step of the ray of every pixel inside the map."""

    def __init__(self, shape, inside, origin, directions, steps, depths, pixelMm):
        self.shape = shape              # (rows, columns) of the map
        self.inside = inside            # (rows, columns) bool: the pixels of the ellipse
        self.origin = origin            # RAS of the brain's centre
        self.directions = directions    # (pixels inside, 3) RAS unit vectors
        self.steps = steps              # (steps,) mm from the centre
        self.depths = depths            # (pixels inside, steps) float16
        self.pixelMm = pixelMm          # the size of a pixel at the centre of the map, mm


def unfoldRays(depth, ijkToRas, center="Scroll", width=UNFOLD_WIDTH, chunk=8192):
    """UnfoldRays of the brain of depth (its depth map on the grid ijkToRas), the map centred on the side center."""
    depth = np.asarray(depth)
    m = np.asarray(ijkToRas, dtype=np.float64)
    axes(m)
    inverse = np.linalg.inv(m)
    voxels = np.argwhere(depth > 0)
    if voxels.size == 0:
        raise ValueError("The brain of the unfolded map is empty.")
    origin = rasOf(voxels.mean(axis=0), m)
    corners = rasOf(np.array([[k, j, i] for k in (0, depth.shape[0] - 1) for j in (0, depth.shape[1] - 1)
                              for i in (0, depth.shape[2] - 1)], np.float64), m)
    reach = float(np.max(np.linalg.norm(corners - origin, axis=1)))
    step = float(min(np.linalg.norm(m[:3, :3], axis=0)))       # a voxel
    steps = np.arange(0.0, reach + step, step)
    front = _centerVector(center)
    if center == "Box":
        grid, inside, perPixel = boxDirections()
        directions = grid[inside]
    elif center == "Poles":
        grid, inside, perPixel = polesDirections()
        directions = grid[inside]
    elif center == "PolesMercator":
        grid, inside, perPixel = polesMercatorDirections()
        directions = grid[inside]
    elif center == "Mercator":
        grid, inside, perPixel = mercatorDirections()
        directions = grid[inside]
    elif front is None:
        grid, inside, perPixel = scrollDirections(width if width != UNFOLD_WIDTH else SCROLL_WIDTH)
        directions = grid[inside]
    else:
        lon, lat, inside = hammerInverse(width)
        perPixel = 4.0 / width          # sqrt(2) map units a radian at the centre, width pixels on 4 sqrt(2) units
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(-front, up)            # the viewer's right, looking at the centre side from outside
        lonIn, latIn = lon[inside], lat[inside]
        directions = (np.cos(latIn)[:, None] * (np.cos(lonIn)[:, None] * front + np.sin(lonIn)[:, None] * right)
                      + np.sin(latIn)[:, None] * up)
    depths = np.zeros((len(directions), len(steps)), np.float16)
    shape = np.array(depth.shape)
    for start in range(0, len(directions), chunk):
        part = directions[start:start + chunk]
        ras = origin + steps[None, :, None] * part[:, None, :]
        ijk = ras @ inverse[:3, :3].T + inverse[:3, 3]
        kji = np.floor(ijk[..., ::-1] + 0.5).astype(np.int64)
        valid = np.all((kji >= 0) & (kji < shape), axis=-1)
        values = np.zeros(valid.shape, np.float32)
        values[valid] = depth[kji[valid, 0], kji[valid, 1], kji[valid, 2]]
        depths[start:start + chunk] = values
    # the scale of the map: a radian the brain's mean radius (its surface voxels) in mm
    surface = voxels[depth[tuple(voxels.T)] <= 1.5]
    radius = float(np.mean(np.linalg.norm(rasOf(surface, m) - origin, axis=1))) if len(surface) else 70.0
    pixelMm = perPixel * radius
    return UnfoldRays(inside.shape, inside, origin, directions, steps, depths, pixelMm)


def unfoldPoints(rays, depthMm):
    """(RAS (rows, columns, 3), hit (rows, columns) bool, (row mm, column mm)) of the unfolded map at depthMm: on
    each ray, coming in from outside, the point where the depth reaches depthMm (between the ray's steps, linearly;
    the surface itself at 0)."""
    threshold = max(float(depthMm), 1e-3)
    reached = rays.depths >= np.float16(threshold)
    count = reached.shape[1]
    last = count - 1 - np.argmax(reached[:, ::-1], axis=1)
    hitIn = reached.any(axis=1)
    rows = np.arange(len(last))
    outer = np.minimum(last + 1, count - 1)
    inside = rays.depths[rows, last].astype(np.float32)
    outside = rays.depths[rows, outer].astype(np.float32)
    step = float(rays.steps[1] - rays.steps[0]) if len(rays.steps) > 1 else 0.0
    drop = inside - outside
    fraction = np.where((outer > last) & (drop > 0), np.clip((inside - threshold) / np.where(drop > 0, drop, 1.0),
                                                               0.0, 1.0), 0.0)
    distance = rays.steps[last] + fraction * step
    rasIn = rays.origin + distance[:, None] * rays.directions
    ras = np.zeros(rays.shape + (3,), np.float64)
    hit = np.zeros(rays.shape, bool)
    ras[rays.inside] = rasIn
    hit[rays.inside] = hitIn
    return ras, hit, (rays.pixelMm, rays.pixelMm)


def unfold(rays, depthMm, ijkToRas):
    """(kji (rows, columns, 3) int, hit (rows, columns) bool, (row mm, column mm)) of the unfolded map at depthMm:
    the voxel of each pixel's point (unfoldPoints)."""
    ras, hit, spacing = unfoldPoints(rays, depthMm)
    inverse = np.linalg.inv(np.asarray(ijkToRas, dtype=np.float64))
    ijk = ras @ inverse[:3, :3].T + inverse[:3, 3]
    kji = np.floor(ijk[..., ::-1] + 0.5).astype(np.int64)
    kji[~hit] = 0
    return kji, hit, spacing
