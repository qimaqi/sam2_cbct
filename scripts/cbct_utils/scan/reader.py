import logging
import os

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import pathlib
from cbct_utils import exceptions
from cbct_utils.scan.meta import group_dicoms_by_size, group_dicoms_by_orientation, read_metadata, get_scanner_info, \
    get_meta_tag, group_dicoms_by_position, _parse_list_tag, is_float, all_close


def validate_by_scan_model(image: sitk.Image):
    known_models = {
        "AUGE SOLIO",
        "SOLIO X"
    }
    scanner_info = get_scanner_info(image)
    model = scanner_info["scanner_model" if scanner_info["scanner_model"] else 'station_name']
    if not (model in known_models):
        _flip_z(image)


def _flip_z(image):
    direction = list(image.GetDirection())
    direction[-1] = -direction[-1]
    image.SetDirection(direction)


def validate_dicom(image: sitk.Image):
    # no need to check direction in this case
    if not any(s == 1.0 for s in image.GetSpacing()):
        return

    spacings = get_meta_tag(image, "PixelSpacing")
    spacings = spacings.split("\\") if spacings else spacings
    if spacings and len(spacings) == 2 and all(is_float(s) for s in spacings):
        x, y = float(spacings[0]), float(spacings[1])
    else:
        x, y, _ = image.GetSpacing()

    # looks like there is no other way to compute z spacing
    thickness = get_meta_tag(image, "SliceThickness")
    if thickness and is_float(thickness):
        new_space = float(thickness)
    else:
        new_space = x

    image.SetSpacing((x, y, new_space))
    # refer to https://dicom.innolitics.com/ciods/ct-image/general-series/00185100
    position = get_meta_tag(image, "PatientPosition")
    orientation = get_meta_tag(image, "PatientOrientation")
    if not position:
        if orientation and "F" not in orientation or not orientation:
            validate_by_scan_model(image)
    elif (orientation and "L" in orientation and "P" in orientation or not orientation) and position == "HFS":
        _flip_z(image)


def validate_sitk(image: sitk.Image):
    """ In case of failed gdcm reader - check direction and spacing
    """
    iop = get_meta_tag(image, "ImageOrientationPatient")
    cosines = _parse_list_tag(iop)
    if cosines and len(cosines) == 6:
        orientation = cosines + [0, 0, 0]
        orientation[6] = orientation[1] * orientation[5] - orientation[2] * orientation[4]
        orientation[7] = orientation[2] * orientation[3] - orientation[0] * orientation[5]
        orientation[8] = orientation[0] * orientation[4] - orientation[1] * orientation[3]

        image.SetDirection(orientation[::3] + orientation[1::3] + orientation[2::3])

    spacings = get_meta_tag(image, "PixelSpacing")
    spacings = spacings.split("\\") if spacings else spacings
    if spacings and len(spacings) == 2 and all(is_float(s) for s in spacings):
        x, y = float(spacings[0]), float(spacings[1])
    else:
        x, y, _ = image.GetSpacing()

    thickness = get_meta_tag(image, "SliceThickness")
    if thickness and is_float(thickness):
        new_space = float(thickness)
    else:
        new_space = x

    image.SetSpacing((x, y, new_space))


def sort_dicoms(dicom_paths, with_meta=False):
    # check whether instance number is presented
    def comp(path):
        if with_meta:
            meta = path[1]
        else:
            meta = read_metadata(path)
        instance_number = get_meta_tag(meta, "InstanceNumber")

        if instance_number and instance_number.isnumeric():
            number = int(instance_number)
        else:
            number = 0

        return number

    return sorted(dicom_paths, key=comp)


def _round_direction(direction):
    return [round(i) for i in direction]


def handle_strange_dicom(reader, sitk_message, dicom_names):
    """ Read normally orientated scan from dicom that contains
        multiple images. If none presented return None
    """
    header_image_error = 'itkImageFileReader.hxx:339'
    rotated_image_error = 'itkImageSeriesReader.hxx:356'
    # specific ITK error handled in this line
    max_dicom_length = 0
    image = None
    dicom_paths = None
    if header_image_error in sitk_message or rotated_image_error in sitk_message:
        groups_by_size = group_dicoms_by_size(dicom_names)
        for size in groups_by_size:
            cur_groups = group_dicoms_by_orientation(groups_by_size[size])
            # can't handle rotated scans
            for group in cur_groups:
                if len(cur_groups[group]) > max_dicom_length:
                    max_dicom_length = len(cur_groups[group])
                    dicom_paths = cur_groups[group]
        dicom_paths = sort_dicoms(dicom_paths)
        reader.SetFileNames(dicom_paths)
        image = reader.Execute()

        direction = _round_direction(image.GetDirection())
        if direction == [0, 0, -1, 1, 0, 0, 0, -1, 0]:
            image.SetDirection([0, 0, 1, 1, 0, 0, 0, -1, 0])
        image = sitk.DICOMOrient(image, 'LPS')

    return image


def _select_scan_dicom_files(reader, path):
    dicom_names = []
    size = 0
    scan_uid = None
    for uid in reader.GetGDCMSeriesIDs(path):
        cur_names = reader.GetGDCMSeriesFileNames(path, seriesID=uid)
        cur_size = sum(map(os.path.getsize, cur_names))
        if cur_size > size:
            dicom_names = cur_names
            size = cur_size
            scan_uid = uid

    return dicom_names, scan_uid


def _invert_intensity(image: sitk.Image):
    interp = get_meta_tag(image, "PhotometricInterpretation")
    if interp:
        # in case of MONOCHROME2 inverse intensities
        if '1' in interp:
            temp_image = image * -1
            for key in image.GetMetaDataKeys():
                value = image.GetMetaData(key)
                temp_image.SetMetaData(key, value)

            image = temp_image
    return image


def dcm_to_image(dir_path: pathlib.Path,
                 *,
                 logger: logging.Logger = logging.getLogger("CBCT"),
                 uid: str = None,
                 drop_duplicates: bool = False,
                 select_scan: bool = True) -> sitk.Image:
    """
    Get image from dicom file/files
    :param dir_path: path to directory with dcm files
    :param logger: logg handler
    :param drop_duplicates: skip duplicates
    :param uid: SeriesID of required DICOM files
    :param select_scan: filter attachments?
    :return: SimpleITK Image with meta tags
    """
    logger.info("Reading DICOM | START")
    if not os.path.isdir(dir_path) and str(dir_path).lower().endswith(".dcm"):
        dcm_files = [dir_path]
        dir_path = os.path.dirname(dir_path)
    else:
        dcm_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.lower().endswith(".dcm")]

    is_series = False

    logger.debug("Total size and file number | {}mb | {}".format(
                                                        round(sum(os.path.getsize(f) for f in dcm_files) / 1024**2, 3),
                                                        len(dcm_files)))
    if len(dcm_files) == 0:
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: No dcm files in directory")
    elif len(dcm_files) == 1:
        reader = sitk.ImageFileReader()
        dcm_file = dcm_files[0]
        reader.SetFileName(dcm_file)
    else:
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(dir_path)
        if len(ids) > 1:
            if select_scan:
                dicom_names, uid = _select_scan_dicom_files(reader, dir_path)
            else:
                uid = uid if uid else ids[0]
                dicom_names = reader.GetGDCMSeriesFileNames(dir_path, uid)

            logger.warning("Found more than 1 dicom series | Using dicom with id {}".format(uid))
        else:
            dicom_names = reader.GetGDCMSeriesFileNames(dir_path)

        if drop_duplicates:
            dicom_names = drop_duplicate_dicom_files(dicom_names)

        if len(dicom_names) == 1:
            reader = sitk.ImageFileReader()
            reader.SetFileName(dicom_names[0])
        else:
            grouped = group_dicoms_by_orientation(dicom_names)

            if len(grouped.keys()) > 1:
                max_len = 0
                for k, v in grouped.items():
                    if len(v) > max_len:
                        dicom_names = v
                        max_len = len(v)
            try:
                dicom_names = ensure_orientation(dicom_names)
                direction = check_direction(dicom_names)
                if not direction:
                    dicom_names = dicom_names[::-1]
            except Exception as e:
                logger.warn("Cannot validate direction | Message: {}".format(e))
                by_sizes = group_dicoms_by_size(dicom_names)
                if len(by_sizes.keys()) > 1:
                    raise exceptions.DicomReadFailedError(
                        "Reading dcm file or directory failed | Message: Multiple scans with same ID"
                    )


            reader.SetFileNames(dicom_names)
            is_series = True

    logger.debug(f"Number of files in DICOM | {len(dcm_files)}")

    try:
        try:
            scan_orig = reader.Execute()
        except RuntimeError as e:
            scan_orig = None
            if is_series:
                scan_orig = handle_strange_dicom(reader, str(e), dicom_names)
            if not scan_orig:
                raise e

        # sitk doesn't load tags from series
        if is_series:
            tagged_image = read_metadata(dicom_names[0])
            for key in tagged_image.GetMetaDataKeys():
                value = tagged_image.GetMetaData(key).encode(errors='ignore').decode()
                scan_orig.SetMetaData(key, value)
            if scan_orig.GetSpacing() == (1.0, 1.0, 1.0):
                validate_sitk(scan_orig)
        else:
            validate_dicom(scan_orig)

        scan_orig = _invert_intensity(scan_orig)

    except RuntimeError as e:
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: {}".format(e)
        )

    scan_info = get_scanner_info(scan_orig)

    logger.info("Reading DICOM | END", extra=scan_info)

    if (scan_orig.GetDimension() == 2 or any(i == 1 for i in scan_orig.GetSize())) and select_scan:
        raise exceptions.DicomReadFailedError(
            "Reading dcm file or directory failed | Message: No scan")

    return scan_orig


def is_rotated(meta):
    # if metadata corresponds to not LFS orientated scan
    orientation = _parse_list_tag(get_meta_tag(meta, "ImageOrientationPatient"))
    rotated_orientations = [[1.0, 0.0, 0.0, 0.0, -1.0, 0.0], # z flipped
                            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]] # x flipped
    return any(all_close(orientation, rotated) for rotated in rotated_orientations)


def ensure_orientation(dicoms):
    orientations = {}
    for file in dicoms:
        orientation = tuple(_parse_list_tag(get_meta_tag(read_metadata(file), "ImageOrientationPatient")))
        for k in orientations.keys():
            if all_close(k, orientation):
                orientations[k].append(file)
                break
        else:
            orientations[orientation] = [file]

    return orientations[max(orientations.keys(), key=lambda k: len(orientations[k]))]


def _get_increasing_index(meta):
    orientation = _parse_list_tag(get_meta_tag(meta, "ImageOrientationPatient"))
    axis = None
    if orientation is None:
        return axis

    if all_close(orientation, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.2) or \
       all_close(orientation, [1.0, 0.0, 0.0, 0.0, -1.0, 0.0], 0.2): # lps oriented
        axis = 2
    if all_close(orientation, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], 0.2) or \
       all_close(orientation, [0.0, 0.0, 1.0, -1.0, 0.0, 0.0], 0.2): # rai oriented
        axis = 1
    if all_close(orientation, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 0.2) or \
       all_close(orientation, [0.0, 1.0, 0.0, 0.0, 0.0, -1.0], 0.2): # ras oriented
        axis = 0

    return axis


def check_direction(dicom_names: list):
    if len(dicom_names) <= 1:
        return True

    a = read_metadata(dicom_names[0])
    b = read_metadata(dicom_names[-1])

    p1 = _parse_list_tag(get_meta_tag(a, "ImagePositionPatient"))
    p2 = _parse_list_tag(get_meta_tag(b, "ImagePositionPatient"))
    is_numbers = lambda x: type(x) is list and len(x) == 3 and all(isinstance(i, (int, float)) for i in x)
    axis = _get_increasing_index(a)
    if is_numbers(p1) and is_numbers(p2):
        if axis is None:
            inds = [i == j for i, j in zip(p1, p2)]
            if inds.count(False) != 1:
                axis = 2
            else:
                axis = inds.index(False)

        return (p2[axis] - p1[axis] > 0) ^ is_rotated(a)

    return True


def drop_duplicate_dicom_files(dicom_paths: list):
    seen = []
    result = []
    for file in dicom_paths:
        meta = read_metadata(file)
        tag = get_meta_tag(meta, "SOPInstanceUID")

        if tag not in seen:
            result.append((file, meta))
            if tag is not None:
                seen.append(tag)

    if len(result) != len(dicom_paths):
        result = sort_dicoms(result, with_meta=True)
        result = [i[0] for i in result]
        direction = check_direction(result)
        if not direction:
            result = result[::-1]
    else:
        result = [i[0] for i in result]

    return result
