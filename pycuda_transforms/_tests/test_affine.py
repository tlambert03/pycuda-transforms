import numpy as np
from pycuda_transforms import transform
from scipy.ndimage import interpolation


def test_affine():
    arr = np.ones((64, 128, 64))
    tmat = np.eye(4)
    tmat[2, 0] = -2
    scipy_ = interpolation.affine_transform(arr, tmat, output_shape=(64, 128, 128))
    ours = transform.affine_transform(arr, tmat, output_shape=(64, 128, 128)).get()
    assert np.allclose(scipy_, ours)


def test_zoom():
    arr = np.zeros((128, 128))
    arr[32:96, 32:96] = 1
    scipy_ = interpolation.zoom(arr, 1.5, order=0)
    ours = transform.zoom(arr, 1.5).get()
    assert np.allclose(scipy_, ours)


def test_rotate():
    arr = np.zeros((128, 128))
    arr[32:96, 32:96] = 1
    scipy_ = interpolation.rotate(arr, -32, order=0, reshape=False)
    ours = transform.rotate(arr, 32).get()
    assert np.allclose(scipy_, ours)


# def compare(funcname, *args, show=False, **kwargs):
#     arr = np.zeros((128, 128))
#     arr[32:96, 32:96] = 1
#     sciresult = getattr(interpolation, funcname)(arr, *args, **kwargs) + 1
#     ourresult = getattr(transform, funcname)(arr, *args, **kwargs).get() + 1
#     try:
#         if not np.allclose(sciresult, ourresult):
#             print("not equal!")
#     except ValueError:
#         print("not the same shape!")
#     if show:
#         viewer = napari.Viewer()
#         viewer.add_image(
#             sciresult,
#             name="scipy result",
#             blending="additive",
#             colormap="magenta",
#             contrast_limits=(0, 2),
#         )
#         viewer.add_image(
#             ourresult,
#             name="pycuda result",
#             blending="additive",
#             colormap="green",
#             contrast_limits=(0, 2),
#         )
#     return sciresult, ourresult
