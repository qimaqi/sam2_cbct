import os
import numpy as np
import SimpleITK as sitk
import nrrd
from cbct_utils.scan.reader import dcm_to_image
from cbct_utils.scan.new_reader import read_dicom
from cbct_utils.scan.meta import get_physical_size
import logging
from typing import Union
import io


def cast(image, sitk_type=sitk.sitkUInt8):
    return sitk.Cast(image, sitk_type)


def save_image(image: sitk.Image, filename: str, compress: bool = True, header={}):
    compression_level = 9 if compress else 1 # 9 is max, 0 makes no sense
    _write_compressed(image, filename, compression_level=compression_level, header=header)


def read_image(filename: Union[str, io.BytesIO],
               uid: str = None,
               drop_duplicates: bool = False,
               select_scan: bool = True,
               engine: str="sitk",
               logger: logging.Logger = logging.getLogger("CBCT")):
    if type(filename) is io.BytesIO or type(filename) is str and (os.path.isdir(filename) or filename.lower().endswith(('.zip', '.dcm'))):
        if engine == "sitk" and type(filename) is str and not filename.endswith(".zip"):
            logger.debug("Using sitk reader")
            return dcm_to_image(filename, uid=uid,
                                drop_duplicates=drop_duplicates,
                                select_scan=select_scan,
                                logger=logger)
        elif engine == "pydicom":
            logger.debug("Using pydicom reader")
            return read_dicom(filename, logger=logger)
        else:
            # TODO: notify mb
            logger.debug("Using pydicom reader")
            return read_dicom(filename, logger=logger)

    return sitk.ReadImage(filename)


def _write_compressed(image: sitk.Image, filename: str, *, compression_level=9, header={}):
    arr = sitk.GetArrayViewFromImage(image).transpose(2, 1, 0)
    nrrd_header = {"space directions": (np.array(image.GetDirection()).reshape((3, 3)) * image.GetSpacing()).T,
                   "space origin": image.GetOrigin(),
                   "space": "left-posterior-superior"}
    nrrd_header.update(header)

    nrrd.write(filename, arr, header=nrrd_header, compression_level=compression_level)


def to_numpy(image: sitk.Image) -> np.ndarray:
    """
    Get volumetric data from sitk image
    :param image: sitk image
    :return: numpy array
    """
    data = sitk.GetArrayFromImage(image).transpose([2, 1, 0])

    x, y, z = map(round, image.GetDirection()[::4])
    data = data[::x, ::y, ::z]

    return data


def create_image(data: np.ndarray, image: sitk.Image) -> sitk.Image:
    """
    Create sitk image using data specified and meta from reference sitk image
    :param data: numpy array
    :param image: sitk image with reference meta
    :return: sitk image
    """
    x, y, z = map(round, image.GetDirection()[::4])
    # if direction is -1 we need manually reverse axis
    data = data[::x, ::y, ::z].transpose([2, 1, 0])

    out_image = sitk.GetImageFromArray(data)

    out_image.SetDirection(image.GetDirection())
    out_image.SetSpacing(image.GetSpacing())
    out_image.SetOrigin(image.GetOrigin())

    return out_image


def change_direction(image: sitk.Image) -> sitk.Image:
    image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return image


def resample(image: sitk.Image,
             dim: tuple,
             size: tuple = None,
             interpolation: int = sitk.sitkNearestNeighbor) -> sitk.Image:
    """
    Resample image to specified resolution
    :param size: needed size
    :param image: sitk image
    :param dim: tuple with target resolution
    :param interpolation: interpolation type
    :return: sitk image
    """
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolation)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(dim)
    min_filter = sitk.MinimumMaximumImageFilter()
    min_filter.Execute(image)
    resample.SetDefaultPixelValue(min_filter.GetMinimum())
    orig_size = np.array(image.GetSize(), dtype=int)
    orig_spacing = image.GetSpacing()
    resample.SetOutputPixelType(sitk.sitkFloat32)
    if not size:
        size = orig_size * (np.divide(orig_spacing, dim))
        # SimpleITK issues #2098, doesn't work lol
        # size = np.ceil(np.round(size, 6)).astype(int)
        size = np.ceil(size).astype(int)
        size = [int(s) for s in size]
    resample.SetSize(size)

    return sitk.Cast(sitk.Round(resample.Execute(image)), image.GetPixelID())
