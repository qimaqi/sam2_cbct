import SimpleITK as sitk
import numpy as np
from scipy import ndimage

from cbct_utils._normalization import transform_by_mapping
from cbct_utils.common import create_image, to_numpy, resample, cast
from cbct_utils.segm.volume import align_segmentation, find_margins

DEFAULT_NORMALIZATION_MAPPING = [1.4210854715202004e-14, 2.827863401354108, 18.829365099643905,
                                 25.140058360576674, 29.130883214754334, 32.329390580355124,
                                 34.100130363454966, 35.61184588336109, 37.64984240523171,
                                 39.20705299497799, 41.544059819118786, 58.32410521607624, 100.00000000000003]


def crop_by_physical_size(image: sitk.Image,
                          physical_size=[156, 156, 156],
                          position=["center", "start", "start"]):
    """
    Crop patch from image with given real size and location 
    from which to start
    :param image: input image
    :param physical_size: list with physical sizes in millimeters
    :param position: list with positions `center` or `start`
    :return: sitk image
    """
    spacing = image.GetSpacing()
    crop_size = [int(ps / sp) for ps, sp in zip(physical_size, spacing)]
    size = image.GetSize()

    start = [0 if pos == "start" else max(0, (sz - cz) // 2) for pos, sz, cz in zip(position, size, crop_size)]

    return image[start[0]:start[0] + crop_size[0],
           start[1]:start[1] + crop_size[1],
           start[2]:start[2] + crop_size[2]]


def crop_image_by_lores(image: sitk.Image,
                        lores: sitk.Image,
                        target_size: tuple = (256,) * 3,
                        target_res: tuple = (0.3,) * 3) -> sitk.Image:
    """
    Crop teeth area from image using teeth center of mass calculated by lowres segmentation.
    Assuming that teeth have label == 3.

    :param image: input image
    :param lores: lores segmentation
    :param target_size: target size of cropped area
    :param target_res: target resolution of cropped area
    :return: sitk image with cropped area of teeth
    """
    lores_cog_vox = ndimage.measurements.center_of_mass(sitk.GetArrayFromImage(lores) == 3)

    print('lo res image:', lores.GetSize(), lores.GetSpacing(), lores.GetOrigin())
    print('lores cog (voxels):', int(lores_cog_vox[0]), int(lores_cog_vox[1]), int(lores_cog_vox[2]))

    lores_cog_x_mm = lores_cog_vox[2] * lores.GetSpacing()[0] + lores.GetDirection()[0] * lores.GetOrigin()[0]
    lores_cog_y_mm = lores_cog_vox[1] * lores.GetSpacing()[1] + lores.GetDirection()[4] * lores.GetOrigin()[1]
    lores_cog_z_mm = lores_cog_vox[0] * lores.GetSpacing()[2] + lores.GetDirection()[8] * lores.GetOrigin()[2]

    print('hires image:', image.GetSize(), image.GetSpacing(), image.GetOrigin())
    hires_cog_x_vox = (lores_cog_x_mm - image.GetOrigin()[0]) / image.GetSpacing()[0]
    hires_cog_y_vox = (lores_cog_y_mm - image.GetOrigin()[1]) / image.GetSpacing()[1]
    hires_cog_z_vox = (lores_cog_z_mm - image.GetOrigin()[2]) / image.GetSpacing()[2]
    print('hires cog (voxels):', int(hires_cog_x_vox), int(hires_cog_y_vox), int(hires_cog_z_vox))

    newimage = resample(image, target_res, interpolation=sitk.sitkNearestNeighbor)

    upadding = tuple(ti // 2 for ti in target_size)
    lpadding = tuple(ti // 2 for ti in target_size)

    newimage = sitk.ConstantPad(newimage, upadding, lpadding, 0.0)

    orig_origin = image.GetOrigin()
    resam_cog_x_vox = int(float((lores_cog_x_mm - newimage.GetDirection()[0] * orig_origin[0]) / target_res[0]))
    resam_cog_y_vox = int(float((lores_cog_y_mm - newimage.GetDirection()[4] * orig_origin[1]) / target_res[1]))
    resam_cog_z_vox = int(float((lores_cog_z_mm - newimage.GetDirection()[8] * orig_origin[2]) / target_res[2]))

    output_image = sitk.RegionOfInterest(newimage, target_size, (resam_cog_x_vox, resam_cog_y_vox, resam_cog_z_vox))

    return output_image


def normalise(image: np.ndarray, mapping: list = DEFAULT_NORMALIZATION_MAPPING) -> np.ndarray:
    """
    Normalise specified image by mapping using histogram standartisation algorithm
    :param image: numpy 3-dimensional image
    :param mapping: list of mapping numbers
    :return: normalised image
    """
    mask = np.ones_like(image, dtype=np.bool)
    im = transform_by_mapping(image, mask, mapping, (0.01, 0.99), type_hist='percentile')

    return im.astype(np.float32)


def normalize_if_needed(image: sitk.Image, threshold=-750):
    filt = sitk.StatisticsImageFilter()
    filt.Execute(image)
    mean = filt.GetMean()
    if mean < threshold:
        arr = to_numpy(image)
        image = create_image(normalise(arr), image)

    return image


def normalize_hist(image: sitk.Image, th=0.999) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(image)
    ns, intensity = np.histogram(arr.reshape(-1), bins=256)

    cutoff_1 = np.ediff1d(ns[:-1]).argmax(axis=0) + 1
    total = np.sum(ns[cutoff_1 + 1:-1])
    cutoff_2 = (np.cumsum(ns[cutoff_1 + 1:].astype(int) / total) > th).argmax() + cutoff_1
    image = sitk.Clamp(image, outputPixelType=sitk.sitkFloat32,
                       lowerBound=float(intensity[cutoff_1]), upperBound=float(intensity[cutoff_2]))

    return sitk.RescaleIntensity(image)


def crop_scan_if_needed(image, segmentation, threshold=2):  # TODO : rewrite on sitk
    """
        Crops scan image depending on segmentation margins and scan padding
    :param image: sitk image
    :param segmentation: sitk image
    :param threshold: how close segmentation could be to border
    :return: same or cropped sitk image
    """
    padding = get_padding(segmentation.GetSize(), (256, 256, 256))
    margins = find_margins(segmentation)
    indexes = np.isclose(-np.ravel(margins), padding, atol=threshold)
    if not indexes.any():
        return image

    sides_close = np.array_split(indexes, 3)
    shapes = image.GetSize() - np.sum(margins, axis=1)
    new_margins = np.array(margins)
    for i, (l, r) in enumerate(sides_close):
        if l:
            new_margins[i][0] = max(0, margins[i][0] - (256 - shapes[i]))
        elif r:
            new_margins[i][1] = max(0, margins[i][1] - (256 - shapes[i]))

    new_image = align_segmentation(image, margins=new_margins, new_shape="optimal")

    return new_image


def get_padding(img_shape: tuple, target_shape: tuple) -> list:
    """
    Calculate padding tuple which can be applied using torch.nn.functional.pad.
    :param img_shape: shape of image
    :param target_shape: target shape
    :return: list with padding
    """
    pad = []

    for dim, min_dim in zip(img_shape, target_shape):
        left_pad = (min_dim - dim) // 2
        right_pad = (min_dim - dim) - left_pad

        pad.extend([left_pad, right_pad])

    return pad


def unpad(data: "torch.tensor", pad: tuple, orig_shape: tuple) -> "torch.tensor":
    """
    Reverse padding operation
    :param data: torch tensor
    :param pad: padding
    :param orig_shape: original shape
    :return: torch tensor of original shape
    """
    if len(data.shape) * 2 > len(pad):
        pad = [0] * (len(data.shape) * 2 - len(pad)) + pad

    in_slices = []
    orig_slices = []
    for c in zip(pad[::2], pad[1::2]):
        s = None if c[0] <= 0 else c[0]
        e = None if c[1] <= 0 else -c[1]
        in_slices.append(slice(s, e))

        s = None if c[0] >= 0 else -c[0]
        e = None if c[1] >= 0 else c[1]
        orig_slices.append(slice(s, e))

    # print(orig_slices, in_slices)

    orig_tensor = np.zeros(orig_shape, dtype=data.dtype)
    orig_tensor[tuple(orig_slices)] = data[tuple(in_slices)]

    return orig_tensor


def detect_blurriness(image: sitk.Image, threshold=0.004) -> bool:
    """
    Returns True if normalized input image has
    blurriness factor less than threshold. Commonly, threshold varies
    from 0.01 to 0.0001
    :param image: input image
    :param threshold: blurriness factor
    :return: bool
    """
    temp_image = crop_by_physical_size(image, [120, 120, 120])
    temp_image = resample(temp_image, (0.6, 0.6, 0.6), interpolation=sitk.sitkLinear)
    temp_image = cast(temp_image, sitk.sitkFloat32)
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(1)
    blured = cast(gaussian.Execute(temp_image), sitk.sitkFloat32)
    filt = sitk.StatisticsImageFilter()
    filt.Execute(temp_image - blured)

    return (filt.GetMean()) < threshold


def detect_overexposure(image: sitk.Image, threshold: int = 600) -> bool:
    """
    Returns True if normalized input image has
    voxels with maximum intensity with volume in mm^3 more than threshold
    :param image: input image
    :param threshold: maximum volume in mm^3
    :return: bool
    """
    temp_image = crop_by_physical_size(image, [120, 120, 120])
    temp_image = resample(temp_image, (0.6, 0.6, 0.6), interpolation=sitk.sitkLinear)
    temp_image = cast(temp_image, sitk.sitkFloat32)
    arr = sitk.GetArrayViewFromImage(temp_image)
    ns, _ = np.histogram(arr)

    return 0.6 ** 3 * ns[-1] > threshold
