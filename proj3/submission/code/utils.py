import numpy as np
import scipy.spatial as sp
import skimage as sk
import itertools

def plot_delaunay(ax, pts):
    ax.triplot(pts[:,0], pts[:,1], sp.Delaunay(pts).simplices)
    ax.plot(pts[:,0], pts[:,1], 'o')

def compute_affine(tri1, tri2):
    ''' given two triangles, computes affine transformation matrix '''
    assert len(tri1) == len(tri2) == 3
    affine_tri1 = np.column_stack([tri1, np.ones(3)]).T
    affine_tri2 = np.column_stack([tri2, np.ones(3)]).T

    barycentric = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 1]])
    # A * [tri1_1, tri1_2, tri1_3] = barycentric
    inv_cob = barycentric @ np.linalg.inv(affine_tri1)

    # A * barycentric = [tri2_1, tri2_2, tri2_3]
    cob = affine_tri2 @ np.linalg.inv(barycentric)

    return (cob @ inv_cob)

def apply_inverse_affine_transformation(A, src, dest, polygon):
    ''' applies the inverse affine A to dest, pulling from src, inside polygon '''
    assert len(polygon) == 2
    assert len(polygon[0]) == len(polygon[1])

    extended = np.vstack((polygon[0], polygon[1], np.ones(len(polygon[0]))))
    inv_polygon = (A @ extended)[:2].astype(int)
    inv_mask = (inv_polygon[0], inv_polygon[1])

    bound = lambda mask, shape: (np.clip(mask[1], 0, shape[0] - 1), np.clip(mask[0], 0, shape[1] - 1), )

    dest[bound(polygon, dest.shape)] = src[bound(inv_mask, src.shape)]


def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    ''' a morph frame from im1 to im2 controlled by fracs '''
    im1_triangles = im1_pts[tri.simplices]
    im2_triangles = im2_pts[tri.simplices]

    # i found that these functions provided a nice pacing for the animatoin
    sigmoid = lambda x: np.e ** x / (1 + np.e ** x)
    zero_one_sigmoid = lambda x: sigmoid((x - 0.5) * 2)
    warp_frac = zero_one_sigmoid(warp_frac)
    dissolve_frac = dissolve_frac ** 2

    warp_pts = (im1_pts * (1. - warp_frac) + im2_pts * warp_frac)
    warp_triangles = warp_pts[tri.simplices]

    im1_warped = np.zeros(im1.shape)
    im2_warped = np.zeros(im2.shape)

    for (tri1, tri2) in zip(warp_triangles, im1_triangles):
        A_inv = compute_affine(tri1, tri2)
        apply_inverse_affine_transformation(A_inv, im1, im1_warped, sk.draw.polygon(tri1[:, 0], tri1[:, 1]))

    for (tri1, tri2) in zip(warp_triangles, im2_triangles):
        A_inv = compute_affine(tri1, tri2)
        apply_inverse_affine_transformation(A_inv, im2, im2_warped, sk.draw.polygon(tri1[:, 0], tri1[:, 1]))

    return (im1_warped * (1. - dissolve_frac) + im2_warped * dissolve_frac)

def parse_asf(im, path):
    ''' parses an asf shapefile from imm_face_db '''
    # line format: <path#> <type> <x rel.> <y rel.> <point#> <connects from> <connects to>
    def parse_row(row):
        row = row.split('\t')
        row = (int(float(row[2]) * im.shape[1]), int(float(row[3]) * im.shape[0]))
        # x, y, from, to
        return row

    with open(path, 'r') as file:
        data = file.read().split('\n')[17:-6]
        pts = np.array([parse_row(row) for row in data])
        corner_pts = list(itertools.product([0, im.shape[1]], [0, im.shape[0]]))
        pts = np.concat([pts, corner_pts])
        return pts, sp.Delaunay(pts).simplices

def warp(im, im_pts, warp_pts, warp_tri):
    ''' warps im to the given triangles '''
    im_triangles = im_pts[warp_tri.simplices]
    warp_triangles = warp_pts[warp_tri.simplices]

    im_warped = np.zeros(im.shape)

    for (tri1, tri2) in zip(warp_triangles, im_triangles):
        A_inv = compute_affine(tri1, tri2)
        apply_inverse_affine_transformation(A_inv, im, im_warped, sk.draw.polygon(tri1[:, 0], tri1[:, 1]))

    return im_warped