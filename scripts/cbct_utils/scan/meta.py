import logging
from collections import defaultdict

import pydicom
import SimpleITK as sitk
from pydicom.datadict import keyword_dict


def sitk_tag_to_pydicom(tag):
    a, b = tag.split('|')
    return int('0x' + a, 16), int('0x' + b, 16)


def parse_tag(tag):
    if isinstance(tag, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS)):
        original_string = tag.original_string
        original_string = original_string.strip()
        return float(original_string) if is_float(original_string) else None
    elif isinstance(tag, pydicom.valuerep.MultiValue):
        return [parse_tag(i) for i in tag]
    elif isinstance(tag, (float, int, str, pydicom.dataset.Dataset)):
        return tag
    else:
        value = str(tag).strip()
        return value if value else None

    return tag


def pydicom_get_tag(ds, tag):
    tag = sitk_tag_to_pydicom(SITK_DICOM_TAGS[tag])
    if tag in ds:
        return parse_tag(ds[tag].value)

    return None


def copy_tags_from_dataset(dataset, image: sitk.Image):
    tag_to_name = {v: k for k, v in SITK_DICOM_TAGS.items()}
    for tag in dataset.values():
        tag_num = int(tag.tag)
        sitk_tag = to_sitk_format(tag_num)
        if sitk_tag in tag_to_name:
            try:
                tag_value = pydicom_get_tag(dataset, tag_to_name[sitk_tag])
                if tag_value is not None and str(tag_value) not in ["", "[]"]:
                    image.SetMetaData(sitk_tag, str(tag_value))
            except:
                ...
                # pydicom can't process some tags values not fully compliant with the standard

    return image


def all_close(first, second, e=0.2):
    for f, s in zip(first, second):
        if abs(f - s) > e:
            return False

    return True


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def _parse_list_tag(tag):
    if tag:
        values = tag.strip().split("\\")
        if len(values) > 1:
            if all([is_float(f) for f in values]):
                return [float(f) for f in values]
            else:
                return values

    return tag


def get_meta_tag(image: sitk.Image, tag: str):
    sitk_tag = SITK_DICOM_TAGS.get(tag, "nope")
    result = None
    if image.HasMetaDataKey(sitk_tag):
        result = image.GetMetaData(sitk_tag)
        if type(result) is str:
            result = result.strip()
            result = result if result else None

    return result


def to_sitk_format(tag: int) -> str:
    second = hex(tag)[2:]
    first = ('0000' + second[:-4])[-4:]
    second = ('000' + second)[-4:]

    return first + '|' + second


SITK_DICOM_TAGS = {name: to_sitk_format(tag) for name, tag in keyword_dict.items()}


def read_metadata(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    return reader


def get_scanner_info(image):
    manufacturer = get_meta_tag(image, "Manufacturer")
    manufacturer_model_name = get_meta_tag(image, "ManufacturerModelName")
    station_name = get_meta_tag(image, "StationName")
    software_version = get_meta_tag(image, "SoftwareVersions")
    software_version2 = get_meta_tag(image, "ImplementationVersionName")

    return {"scanner_manufacturer": manufacturer,
            "scanner_model": manufacturer_model_name,
            "station_name": station_name,
            "software_version": software_version,
            "impl_version": software_version2}


def get_series_length(dir_name, logger: logging.Logger = logging.getLogger("CBCT")):
    '''
        Returns presented and expected count of dcm files
    :param logger: logger instance
    :param dir_name: path to dcm dir
    :return: (presented, expected) number
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir_name)
    if not dicom_names:
        # it occurs, simpleITK cannot read files with names starting with '.'
        # so let's just ignore this error :) CBCT-5925
        return -1, -1

    min_height = 50.0
    first = read_metadata(dicom_names[-1])
    last = read_metadata(dicom_names[0])
    # As it turned out that it may not be presented despite the requirements
    if not first.HasMetaDataKey(SITK_DICOM_TAGS['InstanceNumber']) \
            or not last.HasMetaDataKey(SITK_DICOM_TAGS['InstanceNumber']):
        return -1, -1

    if first.HasMetaDataKey(SITK_DICOM_TAGS['SliceLocation']) and last.HasMetaDataKey(SITK_DICOM_TAGS['SliceLocation']):
        # if it's not float
        try:
            scan_height = abs(float(last.GetMetaData(SITK_DICOM_TAGS['SliceLocation'])) -
                              float(first.GetMetaData(SITK_DICOM_TAGS['SliceLocation'])))
            if scan_height < min_height:
                logger.warning("DICOM scan's height is less than {}mm".format(min_height))
        except ValueError:
            pass

    first = first.GetMetaData(SITK_DICOM_TAGS['InstanceNumber']).strip()
    last = last.GetMetaData(SITK_DICOM_TAGS['InstanceNumber']).strip()
    # in case this tag is empty
    if not first.isnumeric() or not last.isnumeric():
        count = -1
    else:
        count = abs(int(first) - int(last)) + 1

    return len(dicom_names), count


def group_dicoms_by_direction(dcms_paths):
    """ In case of multiple image in dicom
    group DCMs by direction"""
    dif_dcms = defaultdict(lambda: [])
    for dcm in dcms_paths:
        md = read_metadata(dcm)
        rows = md.GetDirection()
        dif_dcms[rows].append(dcm)

    return dif_dcms


def group_dicoms_by_size(dcms_paths):
    """ In case of multiple image in dicom
    group DCMs by size"""
    dif_dcms = defaultdict(lambda: [])
    for dcm in dcms_paths:
        md = read_metadata(dcm)
        rows = md.GetMetaData(SITK_DICOM_TAGS['Rows'])
        cols = md.GetMetaData(SITK_DICOM_TAGS['Columns'])
        dif_dcms[rows + '_' + cols].append(dcm)

    return dif_dcms


def group_dicoms_by_position(dicom_names):
    positions = defaultdict(lambda: [])
    for name in dicom_names:
        md = read_metadata(name)
        pos = _parse_list_tag(get_meta_tag(md, "ImagePositionPatient"))
        if pos:
            xz = tuple([pos[0], None, pos[2]])
            xy = tuple([pos[0], pos[1], None])
            yz = tuple([None, pos[1], pos[2]])
            positions[xz].append(name)
            positions[xy].append(name)
            positions[yz].append(name)
        else:
            positions[None].append(name)

    grouped = {k:v for k, v in positions.items() if len(v) > 1}
    all_used_files = set([j for i in grouped.values() for j in i])
    while len(all_used_files) != len(dicom_names):
        for k, v in positions.items():
            if len(v) == 1:
                if v[0] not in all_used_files:
                    grouped[k] = v
                    all_used_files.add(v[0])

    return grouped


def group_dicoms_by_orientation(dicom_names):
    orientations = defaultdict(lambda: [])
    for name in dicom_names:
        md = read_metadata(name)
        cur_or = _parse_list_tag(get_meta_tag(md, "ImageOrientationPatient"))
        if cur_or and len(cur_or) == 6:
            cur_or = tuple(cur_or)
            for orientation in orientations:
                if all_close(orientation, cur_or):
                    orientations[orientation].append(name)
            else:
                orientations[cur_or].append(name)
        else:
            orientations[None].append(name)

    grouped = {k:v for k, v in orientations.items() if len(v) > 1}
    all_used_files = set([j for i in grouped.values() for j in i])
    while len(all_used_files) != len(dicom_names):
        for k, v in orientations.items():
            if len(v) == 1:
                if v[0] not in all_used_files:
                    grouped[k] = v
                    all_used_files.add(v[0])

    return grouped


def get_physical_size(image: sitk.Image):
    """
    Computes real size
    """
    spacing = image.GetSpacing()
    size = image.GetSize()

    return [sp * si for sp, si in zip(spacing, size)]
