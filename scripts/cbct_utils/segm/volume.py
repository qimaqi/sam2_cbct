import numpy as np
import SimpleITK as sitk
from cbct_utils.common import to_numpy


def find_margin(arr,
                axis=0,
                high=None,
                threshold=10):
    '''
        Finds first occurance of segmentation
    :param arr: 3d numpy array
    :param high: max index
    :param axis: in which axis search
    :param threshold: max count of nonzero voxels that don't belong to segmentation
    :return: count of voxels from 0 to segmentation border
    '''
    low = 0
    if not high:
        high = arr.shape[axis] // 2
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        if np.count_nonzero(arr.take(mid, axis)) < threshold:
            low = mid + 1
        else:
            high = mid - 1

    return mid



def find_margins(image):
    '''
        Finds margins of segmentation
    :param image: sitk image
    :return: ((left, right), (front, back), (bottom, top)) - margins
    '''
    arr = to_numpy(image)
    x = [find_margin(arr, axis=0), find_margin(arr[::-1], axis=0)]
    y = [find_margin(arr, axis=1), find_margin(arr[::, ::-1], axis=1)]
    z = [find_margin(arr, axis=2), find_margin(arr[..., ::-1], axis=2)]

    return x, y, z


def align_segmentation(image,
                       margins=None,
                       new_shape="same",
                       seg_size=[256, 256, 256]):
    """
        Detects segmentation and aligns it to input image shape
    :param image: input image
    :param margins: ((left, right), (front, back), (bottom, top)) margins or None
    :param new_shape: "same", "optimal" or [x, y, z] - new image shape
    :param seg_size: size of segmentation
    :return: sitk image with moved segmentation
    """
    x, y, z = margins if margins is not None else find_margins(image)

    arr = to_numpy(image)
    shape = [arr.shape[i] - sum(margins) for i, margins in enumerate((x, y, z))]
    origin = image.GetOrigin()
    direction = image.GetDirection()
    spacing = image.GetSpacing()
    if new_shape == "same":
        new_array = np.zeros_like(arr, dtype=arr.dtype)
        x[0], y[0], z[0] = [max(0, ax[0] - (seg_size[i] - shape[i]) // 2) for i, ax in enumerate((x, y, z))]
        shape = [min(size - ax, cur_size) for size, cur_size, ax in zip(arr.shape, seg_size, (x[0], y[0], z[0]))]
        new_origin = (x[0], y[0], z[0] if direction[8] > 0 else -max(0, arr.shape[2] - z[0] - shape[2]))
    elif new_shape == "optimal":
        new_array = np.empty(shape, dtype=arr.dtype)
        new_origin = (x[0], y[0], z[0] if direction[8] > 0 else -z[1])
    else:
        new_array = np.zeros(new_shape, dtype=arr.dtype)
        shape = new_shape

    new_array[:shape[0],
              :shape[1],
               slice(None, shape[2]) if direction[8] > 0
          else slice(-shape[2], None)] = arr[x[0]:x[0] + shape[0],
                                             y[0]:y[0] + shape[1],
                                             z[0]:z[0] + shape[2]]

    x_, y_, z_ = map(round, image.GetDirection()[::4])
    new_array = new_array[::x_, ::y_, ::z_].transpose([2, 1, 0])

    new_image = sitk.GetImageFromArray(new_array)

    new_origin = [coord * spacing[i] + origin[i] for i, coord in enumerate(new_origin)]

    new_image.SetSpacing(spacing)
    new_image.SetOrigin(new_origin)
    new_image.SetDirection(direction)

    return new_image

