import io
import logging
import os
import pathlib
import zipfile
from collections import defaultdict, namedtuple
from typing import Union

import numpy as np
import pydicom
import SimpleITK as sitk
from cbct_utils import exceptions
from cbct_utils.scan.meta import (SITK_DICOM_TAGS, all_close,
                                  copy_tags_from_dataset, get_physical_size,
                                  get_scanner_info, is_float, parse_tag,
                                  pydicom_get_tag, sitk_tag_to_pydicom)
from pydicom.util import fixer


MIN_ALLOWED_SIZE = 30.0
MAX_ALLOWED_SPACING = 2.0


def select_max_size_series(files):
    uids_sizes = defaultdict(int)
    for dcm in files:
        uids_sizes[pydicom_get_tag(dcm.dataset, "SeriesInstanceUID")] += dcm.size
    max_size = 0
    max_uid = None
    for uid in uids_sizes:
        if uids_sizes[uid] > max_size:
            max_size, max_uid = uids_sizes[uid], uid

    return [file for file in files if pydicom_get_tag(file.dataset, "SeriesInstanceUID") == max_uid]


def drop_duplicate_dicoms(files):
    uids_files = defaultdict(list)
    for dcm in files:
        uids_files[pydicom_get_tag(dcm.dataset, "SOPInstanceUID")].append(dcm)

    for uid in uids_files:
        if len(uids_files[uid]) > 1:
            correct_file = uids_files[uid][0]
            for file in uids_files[uid][1:]:
                if is_the_same_dicom(correct_file.dataset, file.dataset):
                    files.remove(file)

    return files


def is_the_same_dicom(dcm1, dcm2):
    if pydicom_get_tag(dcm1, "SOPInstanceUID") != pydicom_get_tag(dcm2, "SOPInstanceUID"):
        return False
    # 15999192 - for name, 10851727 - for bits.
    can_be_different = [sitk_tag_to_pydicom(SITK_DICOM_TAGS[tag_name]) for tag_name in
                        ["InstanceCreationTime", "InstanceCreationDate", "ContentDate", "ContentTime",
                         "FrameOfReferenceUID", "PatientName", "HighBit", "BitsStored"]]
    different_tags = []
    for tag in dcm1:
        cur_tag = tag.tag.group, tag.tag.elem
        if cur_tag not in dcm2 or dcm2[cur_tag] != tag:
            if cur_tag not in can_be_different and not tag.is_private:
                if dcm2[cur_tag] != tag and tag.VR == "SQ":  # 80429146
                    # different_tags.extend(parse_nested_tags(dcm1))
                    # just ignore...
                    pass
                else:
                    different_tags.append(tag)
    if not different_tags:
        return True

    return False


def guess_transfer_syntax(dicom_data):
    is_upper = lambda b: chr(b).isupper()
    result = pydicom.uid.ImplicitVRLittleEndian
    first_tag = dicom_data[:8]
    if is_upper(first_tag[4]) and is_upper(first_tag[5]):
        result = pydicom.uid.ExplicitVRLittleEndian
        if first_tag[0] < first_tag[1]:
            result = pydicom.uid.ExplicitVRBigEndian

    return result


def fix_pydicom_dataset(dataset):
    values = list(dataset.values())
    for value in values: # 17696858
        if value.VR == 'IS' and (value.value is None or b'.' in value.value):
            dataset.pop(value.tag)

    return dataset


def read_dicom_header(filename):
    dataset = pydicom.dcmread(filename, force=True, stop_before_pixels=True)

    return fix_pydicom_dataset(dataset)


def read_dicom_pixel_data_from_bytes(data):
    from pydicom.pixel_data_handlers import numpy_handler, gdcm_handler, pillow_handler, jpeg_ls_handler, pylibjpeg_handler, rle_handler
    pydicom.config.pixel_data_handlers = [gdcm_handler,
                                          numpy_handler,
                                          pylibjpeg_handler,
                                          pillow_handler,
                                          jpeg_ls_handler,
                                          rle_handler]
    #     with open(filename, "rb") as file:
    dataset = pydicom.dcmread(io.BytesIO(data), force=True, stop_before_pixels=False)
    data_size = (dataset.Rows, dataset.Columns)
    if "TransferSyntaxUID" not in dataset.file_meta: # no header with transfer syntax
        transfer_syntax = guess_transfer_syntax(data)
        dataset.file_meta.TransferSyntaxUID = transfer_syntax
    try:
        # dataset.decompress("pylibjpeg")
        data = dataset.pixel_array
    except (ValueError, AttributeError): # in case of failed decoding (11880038)
        if "PixelData" in dataset and dataset.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.91": # JPEG 2000 Image Compression
            data = pydicom.pixel_data_handlers.pillow_handler.get_pixeldata(dataset).reshape(data_size) # https://github.com/pydicom/pylibjpeg/issues/51
        else:
            # if fails anyway - return zeros
            data = np.zeros(data_size, dtype=np.uint16) # TODO : somehow define dtype from dataset

    return data


def select_by_orientation(files):
    orientations = defaultdict(list)
    for dcm in files:
        cur_or = pydicom_get_tag(dcm.dataset, "ImageOrientationPatient")
        if cur_or and len(cur_or) == 6:
            cur_or = tuple(cur_or)
            for orientation in orientations:
                if orientation is not None and all_close(orientation, cur_or):
                    orientations[orientation].append(dcm)
                    break
            else:
                orientations[cur_or].append(dcm)
        else:
            orientations[None].append(dcm)

    return max(orientations.values(), key=len)


def select_by_implementation_version_name(files):
    impl_versions = defaultdict(list)
    for dcm in files:
        if dcm.dataset.file_meta is not None:
            cur_vers = pydicom_get_tag(dcm.dataset.file_meta, "ImplementationVersionName")
        else:
            cur_vers = None

        impl_versions[cur_vers].append(dcm)

    return max(impl_versions.values(), key=lambda fs: sum(i.size for i in fs))


def order_dicoms_for_reading(files):
    orientation = pydicom_get_tag(files[0].dataset, "ImageOrientationPatient")
    if orientation and type(orientation) is list and len(orientation) == 6:
        index = 3 - (np.argmax(np.abs(orientation[:3])) + np.argmax(np.abs(orientation[3:6])))
        patient_pos = lambda elem: (float(s[index]) if type((s:=pydicom_get_tag(elem.dataset, "ImagePositionPatient"))) is list and is_float(s[index]) else None)
        if all(patient_pos(file) is not None for file in files):
            return sorted(files, key=patient_pos)

    slice_loc = lambda elem: (float(s) if (s:=pydicom_get_tag(elem.dataset, "SliceLocation")) is not None and is_float(s) else None)
    if all(slice_loc(file) is not None for file in files):
        # 80421217 duplicate slice location
        if len(set([pydicom_get_tag(elem.dataset, "SliceLocation") for elem in files])) == len(files):
            return sorted(files, key=slice_loc)

    # InstanceNumber may be missleading. DO NOT USE IT 21035450
    inst_number = lambda elem: (float(s) if (s:=pydicom_get_tag(elem.dataset, "InstanceNumber")) is not None and is_float(s) else None)
    if all(inst_number(file) is not None for file in files):
        return sorted(files, key=inst_number)

    return files


def check_orientation(orientation_matrix, origin, second_slice_position):
    # check if orientation corresponds to slice reading order
    res = np.subtract(second_slice_position, origin) @ np.linalg.inv(orientation_matrix)
    if res[-1] < 0:
        return False

    return True


def get_increasing_index(dcm):
    axis = 2
    orientation = pydicom_get_tag(dcm, "ImageOrientationPatient")
    if orientation and type(orientation) is list and len(orientation) == 6:
        axis = 3 - (np.argmax(np.abs(orientation[:3])) + np.argmax(np.abs(orientation[3:6])))

    return axis


def select_by_size(files):
    sizes = defaultdict(list)
    for dcm in files:
        cur_size = dcm.dataset.Columns, dcm.dataset.Rows
        sizes[cur_size].append(dcm)
    return max(sizes.values(), key=len)


def guess_thickness(datasets, pixel_spacing=None, thickness=None):
    if datasets:
        if len(datasets) > 1:
            thicknesses = []
            for i in range(1, len(datasets)):
                a = pydicom_get_tag(datasets[i - 1], "ImagePositionPatient")
                b = pydicom_get_tag(datasets[i], "ImagePositionPatient")
                if a is not None and b is not None:
                    dist = np.linalg.norm(np.array(a) - b)
                    thicknesses.append(dist)

            return np.mean(thicknesses)

        pixel_spacing = pydicom_get_tag(datasets[0], "PixelSpacing")
    if thickness:
        return thickness
    if pixel_spacing:
        return round(np.mean(pixel_spacing), 3)

    return 0.3


def guess_spacing(dataset, thickness):
    # in case if there are no spacings in meta
    if dataset.Columns == dataset.Rows:
        return [thickness, thickness]

    typical_spacings = [0.11, 0.15, 0.16, 0.2, 0.25, 0.3, 0.6, thickness]
    spacings = []
    for cur_size in [dataset.Rows, dataset.Columns]:
        min_diff = 100
        ratio = 8.2 / cur_size
        cur_spacing = 1.0
        for spacing in typical_spacings:
            if abs(ratio - spacing) < min_diff:
                min_diff = abs(ratio - spacing)
                cur_spacing = spacing
        spacings.append(cur_spacing)

    return spacings


def parse_nested_tags(dataset):
    tags = []
    if dataset is not None:
        for tag in dataset:
            if type(tag) is pydicom.Dataset:
                tags.extend(parse_nested_tags(tag))
            elif type(tag.value) is pydicom.sequence.Sequence:
                if len(tag.value) == 1:
                    tags.extend(parse_nested_tags(tag.value[0]))
            else:
                tags.append(tag)

    ds = pydicom.Dataset()
    for tag in tags:
        ds[tag.tag] = tag

    return ds


def merge_datasets(dataset, *datasets):
    for ds in datasets:
        for tag in ds:
            if tag.tag not in dataset:
                dataset[tag.tag] = tag
    return dataset

# TODO : find models
def flip_by_model_scanner(dataset):
    known_models = ["AUGE SOLIO", "SOLIO X"]
    model = pydicom_get_tag(dataset, "StationName")
    if model in known_models:
        return False

    return True


def apply_lut_to_image(image: sitk.Image, ds: pydicom.Dataset) -> sitk.Image:
    slope = pydicom_get_tag(ds, "RescaleSlope")
    intercept = pydicom_get_tag(ds, "RescaleIntercept")
    if isinstance(slope, (int, float)) and isinstance(intercept, (int, float)):
        image = sitk.Cast(image, sitk.sitkFloat32) * float(slope) + float(intercept)
        # check if slope is real 64255803
        if slope % 1 or intercept % 1:
            return image

    return sitk.Cast(image, sitk.sitkInt32)


def get_sitk_pixel_type(filename):
    """Returns pixel component type from GDCM reader"""
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)

    reader.ReadImageInformation()
    pixel_type = reader.GetPixelIDValue()
    return pixel_type if pixel_type != sitk.sitkUnknown else sitk.sitkInt32


def read_dicom(path: Union[pathlib.Path, str, io.BytesIO],
               downsample=False,
               downsample_spacing=0.3,
               logger=logging):
    logger.info("Reading DICOM | START")
    zipped = True
    if isinstance(path, (pathlib.Path, str)):
        path = str(path)
        if path.rsplit('.')[-1] != 'zip':
            zipped = False
            dcm_files = []
            if not os.path.isdir(path) and str(path).lower().endswith(".dcm"):
                dcm_files = [path]
            elif os.path.isdir(path):
                dcm_files = [os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith(".dcm")]

            if len(dcm_files) == 0:
                raise exceptions.DicomReadFailedError(
                    "Reading dcm file or directory failed | Message: No dcm files in directory")

    files = []
    nt = namedtuple("DCMFile", ["filename", "size", "dataset"])
    fixer.fix_mismatch()
    if zipped:
        try:
            zf = zipfile.ZipFile(path)
            for file in zf.filelist:
                with zf.open(file) as f:
                    files.append(nt(file.filename, file.file_size, read_dicom_header(f)))
        except Exception as e:
            raise exceptions.ExtractionFailedError(
                "Can't extract zip archive | Message: {}".format(e)
            )
    else:
        for file in dcm_files:
            files.append(nt(file, os.path.getsize(file), read_dicom_header(file)))
    prev_len = len(files)
    logger.debug("Total size and file number | {}mb | {}".format(round(sum(i.size for i in files) / (1024)**2, 3), prev_len))
    files = select_max_size_series(files)
    if len(files) != prev_len:
        logger.warning("Found more than 1 dicom series | Using dicom with id {}".format(pydicom_get_tag(files[0].dataset, "SeriesInstanceUID")))
    prev_len = len(files)
    files = drop_duplicate_dicoms(files)
    if len(files) != prev_len:
        logger.debug("Found DICOM files duplicates")
    prev_len = len(files)
    files = select_by_orientation(files)
    if len(files) != prev_len:
        logger.debug("Multivolume scan with different orientations")
    prev_len = len(files)
    files = select_by_size(files)
    if len(files) != prev_len:
        logger.debug("Multivolume scan with different sizes")
    files = select_by_implementation_version_name(files)
    if len(files) != prev_len:
        logger.debug("Single-file and Multi-file DICOMs with the same SeriesUID")
    files = order_dicoms_for_reading(files)
    logger.debug(f"Number of files in DICOM | {len(files)}")
    #     files = flip_if_needed(files)
    for_meta = files[0].dataset
    filenames_in_correct_order = [file.filename for file in files]

    # if not series - read nested tags
    is_series = True
    per_frame_datasets = None
    if len(filenames_in_correct_order) == 1:
        is_series = False
        shared = pydicom_get_tag(for_meta, "SharedFunctionalGroupsSequence")
        if shared is not None:
            for_meta = merge_datasets(for_meta, parse_nested_tags(shared))
        per_frame = pydicom_get_tag(for_meta, "PerFrameFunctionalGroupsSequence")
        if per_frame is not None:
            per_frame_datasets = [parse_nested_tags(pf) for pf in per_frame]
            for_meta = merge_datasets(for_meta, *per_frame_datasets)
        else:
            per_frame = []

    # read important for DICOM orientation tags
    pixel_spacing = pydicom_get_tag(for_meta, "PixelSpacing")
    thickness = pydicom_get_tag(for_meta, "SliceThickness")
    origin = pydicom_get_tag(for_meta, "ImagePositionPatient")
    patient_orientation = pydicom_get_tag(for_meta, "PatientOrientation")
    patient_position = pydicom_get_tag(for_meta, "PatientPosition")

    # slice thickness may be incorrect
    thickness = guess_thickness([file.dataset for file in files] if is_series else per_frame_datasets,
                                thickness=thickness)
    if pixel_spacing is None:
        pixel_spacing = guess_spacing(for_meta, thickness)
    spacing = pixel_spacing + [thickness]
    index = get_increasing_index(for_meta)

    # determine principal axis of DICOM
    can_determine_orientation = False
    second_slice_position = None
    flip = False
    if origin:
        if not is_series:
            if per_frame_datasets is not None:
                last_slice_position = pydicom_get_tag(per_frame_datasets[-1], "ImagePositionPatient")
                if last_slice_position is not None:
                    if origin[index] > last_slice_position[index]:
                        flip = True
                    else:
                        flip = False
                    can_determine_orientation = True
                if len(per_frame_datasets) > 1:
                    second_slice_position = pydicom_get_tag(per_frame_datasets[1], "ImagePositionPatient")
        else:
            last_slice_position = pydicom_get_tag(files[-1].dataset, "ImagePositionPatient")
            if last_slice_position is not None:
                if origin[index] > last_slice_position[index]:
                    flip = True
                elif patient_orientation in [["P", "F"]]: # there also might be other orientations
                    flip = True
                else:
                    flip = False
                can_determine_orientation = True
                if len(files) > 1:
                    second_slice_position = pydicom_get_tag(files[1].dataset, "ImagePositionPatient")

    if not is_series and not can_determine_orientation:
        if not patient_position:
            logger.warning("Cannot determine orientation")
            flip = flip_by_model_scanner(for_meta)
        elif (patient_orientation == ['L', 'P'] or patient_orientation is None) and patient_position == "HFS":
            flip = True

    cosines = pydicom_get_tag(for_meta, "ImageOrientationPatient")
    orientation = None
    if cosines:
        orientation = cosines + [0, 0, 0]
        orientation[6] = (orientation[1] * orientation[5] - orientation[2] * orientation[4]) * (-1 if flip else 1)
        orientation[7] = (orientation[2] * orientation[3] - orientation[0] * orientation[5]) * (-1 if flip else 1)
        orientation[8] = (orientation[0] * orientation[4] - orientation[1] * orientation[3]) * (-1 if flip else 1)
        orientation_matrix = np.array(orientation).reshape(3, 3)
        if not flip and can_determine_orientation and second_slice_position is not None and origin is not None:
            flip = not check_orientation(orientation_matrix, origin, second_slice_position)
            # could be problems with scans rotated on 30+ degrees.
            if flip:
                orientation_matrix[-1, ::] *= -1

        orientation = orientation_matrix.T.ravel().tolist()
    # compute new_spacing and strides for downsampling
    strides = [slice(None, None, max(int(round(downsample_spacing / i, 1)), 1)) if downsample else slice(None) for i in spacing]
    if downsample and any(i not in [slice(None), slice(None, None, 1)] for i in strides):
        logger.debug("Downsampling DICOM: {} -> {}".format(spacing,
                                                           spacing:=[i * max(int(round(downsample_spacing / i, 1)), 1) for i in spacing]))
    # read pixel data from dcm files
    filenames_in_correct_order = filenames_in_correct_order[strides[2]]
    try:
        if not is_series:
            if zipped:
                img = sitk.GetImageFromArray(read_dicom_pixel_data_from_bytes(
                    zf.read(
                        filenames_in_correct_order[0]))[strides[0], strides[1], strides[2]])
                zf.close()
            else:
                with open(filenames_in_correct_order[0], "rb") as file:
                    img = sitk.GetImageFromArray(read_dicom_pixel_data_from_bytes(file.read())[strides[0],
                                                                                               strides[1],
                                                                                               strides[2]])
        else:
            img = []
            if zipped:
                for file in filenames_in_correct_order:
                    img.append(sitk.GetImageFromArray(read_dicom_pixel_data_from_bytes(zf.read(file))[strides[0],
                                                                                                      strides[1]]))
                zf.close()
            else:
                for file in filenames_in_correct_order:
                    with open(file, "rb") as file:
                        img.append(sitk.GetImageFromArray(read_dicom_pixel_data_from_bytes(file.read())[strides[0],
                                                                                                        strides[1]]))
            img = sitk.JoinSeries(img)
    except Exception as e:
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: {}".format(e)
        )
    # construct sitk image
    if orientation:
        img.SetDirection(orientation)
    if origin:
        img.SetOrigin(origin)
    if spacing:
        img.SetSpacing(spacing)

    if img.GetDimension() == 2 or any(i == 1 for i in img.GetSize()):
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: No scan")

    min_height = 50.0
    scan_height = min(get_physical_size(img))
    if scan_height < min_height:
        logger.warning("DICOM scan's height is less than {}mm".format(min_height))
    if scan_height < MIN_ALLOWED_SIZE or max(img.GetSpacing()) > MAX_ALLOWED_SPACING:
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: No scan")
    # handle RGB (for some reasons...)
    if type(img[0, 0, 0]) is tuple:
        img = sitk.VectorIndexSelectionCast(img, 0)

    img = apply_lut_to_image(img, for_meta)
    img = copy_tags_from_dataset(for_meta, img)
    scan_info = get_scanner_info(img)
    # idk why, but GDCM sometimes casts int16 to int32, and it affects further processing... so let's
    # just do it like in simpleITK.
    # if not zipped:
    #     img = sitk.Cast(img, get_sitk_pixel_type(filenames_in_correct_order[0]))

    logger.info("Reading DICOM | END", extra=scan_info)
    return img
