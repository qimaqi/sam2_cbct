import SimpleITK as sitk
import numpy as np
from scipy import ndimage

from scipy.spatial import KDTree
from cbct_utils.common import cast, to_numpy, create_image
import logging


def fill_holes(image: sitk.Image, margin: int = 20) -> sitk.Image:
    # TODO : limit highest point of lower jaw
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(image)
    x, y, z = image.TransformPhysicalPointToIndex(f.GetCentroid(3))
    size = image.GetSize()
    t = image[max(x - 200, 0):min(x + 200, size[0]), :y, max(z - 150, 0):z]
    f.Execute(t)

    arr = to_numpy(t)
    for i, j, k in np.ndindex(*arr.shape[:2], f.GetRegion(2)[-1] - margin):
        if arr[i, j, k] == 3:
            temp = arr[i - 1:i + 1, j - 1: j + 1, k - 1: k + 1]
            arr[i - 1:i + 1, j - 1: j + 1, k - 1: k + 1] = np.where(temp == 0, 2, temp)

    image[max(x - 200, 0):min(x + 200, size[0]), :y, max(z - 150, 0):z] = create_image(arr, t)

    return image


def crop_foreground(image: sitk.Image, return_coords: bool = False, label: int = None) -> sitk.Image:
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(image if label else image != 0)
    st, sizes = np.array_split(f.GetBoundingBox(label if label else 1), 2)

    result = image[st[0]:st[0] + sizes[0], st[1]:st[1] + sizes[1], st[2]:st[2] + sizes[2]]

    if return_coords:
        return result, st

    return result


def validate_bones(image: sitk.Image) -> sitk.Image:
    # TODO : rewrite on pure sitk
    arr = to_numpy(image)
    cog_vox = [round(i) for i in ndimage.measurements.center_of_mass(arr == 3)]
    arr[..., :cog_vox[2]] = np.where(arr[..., :cog_vox[2]] == 1, 2, arr[..., :cog_vox[2]])

    return create_image(arr, image)


def match_images(image_from: sitk.Image, image_to: sitk.Image, interpolation=sitk.sitkNearestNeighbor):
    """
    Resamples/Crops/Resizes/Rotates image_from
    to make it same as image_to
    :param seg: image to transform
    :param seg: original
    :return: Corresponding image
    """
    
    return sitk.Resample(image_from,
                         image_to.GetSize(),
                         sitk.Transform(),
                         interpolation,
                         image_to.GetOrigin(),
                         image_to.GetSpacing(),
                         image_to.GetDirection())


def crop_by_cog(image: sitk.Image, label: int, roi_size=[256, 256, 256]):
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(image)
    if label not in filt.GetLabels():
        filt.Execute(image != 0)
        label = 1
    center = image.TransformPhysicalPointToIndex(filt.GetCentroid(label))

    return image[max(0, center[0] - roi_size[0] // 2):center[0] + roi_size[0] // 2,
           max(0, center[1] - roi_size[1] // 2):center[1] + roi_size[1] // 2,
           max(0, center[2] - roi_size[2] // 2):center[2] + roi_size[2] // 2]


def get_spacing(image: sitk.Image) -> float:
    return np.mean(image.GetSpacing()).item()


def _get_components_stats(components):
    label_shape = sitk.LabelShapeStatisticsImageFilter()
    label_shape.Execute(components)

    return sorted([(i, label_shape.GetNumberOfPixels(i)) for i in label_shape.GetLabels()],
                  key=lambda z: -z[1])


def filtrate_mc(mc_preds: sitk.Image, min_tooth_volume=3000, to_merge_label=77, merge=True) -> sitk.Image:
    label_shape = sitk.LabelShapeStatisticsImageFilter()

    # filter artifacts
    comp = sitk.ConnectedComponent(mc_preds > 0)
    r = _get_components_stats(comp)
    mask = sitk.ChangeLabel(comp, {i[0]: 1 if i[1] > min_tooth_volume else 0 for i in r})
    mc_preds *= cast(mask)

    # merge parts
    main_comp = mc_preds * 0

    label_shape.Execute(mc_preds)

    for label in label_shape.GetLabels():
        comp = sitk.ConnectedComponent(mc_preds == label)
        r = _get_components_stats(comp)
        to_filt = {r[0][0]: label if r[0][1] > min_tooth_volume else to_merge_label}
        to_filt.update({i[0]: to_merge_label for i in r[1:]})
        comp = sitk.ChangeLabel(comp, to_filt)
        main_comp += cast(comp)

    if not merge:
        main_comp = sitk.ChangeLabel(main_comp, {to_merge_label: 0})
        return main_comp
    # check if segment connected to only tooth
    comp = sitk.ConnectedComponent(main_comp == to_merge_label)
    r = _get_components_stats(comp)
    comp = sitk.ChangeLabel(comp, {i[0]: i[0] if i[1] > min_tooth_volume else 0 for i in r})
    label_shape.Execute(main_comp)
    pairs = {}
    not_one = set()
    for label_main in label_shape.GetLabels():
        if label == to_merge_label:
            continue
        label_shape.Execute(comp)
        for label_comp in label_shape.GetLabels():
            temp_comp = sitk.ConnectedComponent(cast(main_comp == label_main) + cast(comp == label_comp))
            label_shape.Execute(temp_comp)
            if label_shape.GetNumberOfLabels() == 1:
                if label_comp in pairs:
                    not_one.add(label_comp)
                pairs[label_comp] = label_main

    to_filt = {}
    label_shape.Execute(comp)
    for k in pairs:
        if k not in not_one:
            to_filt[k] = pairs[k]
    for label in label_shape.GetLabels():
        if label not in to_filt:
            to_filt[label] = 0

    not_used = cast(main_comp == to_merge_label) - (cast(sitk.ChangeLabel(comp, to_filt)) > 0)
    main_comp = main_comp - cast(main_comp == to_merge_label) * to_merge_label + cast(
        sitk.ChangeLabel(comp, to_filt))

    # check if segment connected to more than one tooth
    arr = to_numpy(main_comp)
    coords = np.array(np.nonzero(arr)).T
    kdtree = KDTree(coords)
    not_used_coords = np.array(np.nonzero(to_numpy(not_used))).T
    closest = kdtree.query(not_used_coords)

    arr[not_used_coords[:, 0],
        not_used_coords[:, 1],
        not_used_coords[:, 2]] = arr[coords[closest[1]][:, 0],
                                     coords[closest[1]][:, 1],
                                     coords[closest[1]][:, 2]]

    # merge big segments
    # filt = sitk.LabelShapeStatisticsImageFilter()
    # filt.Execute(out_image)
    # for label in filt.GetLabels()[2:]:
    #     comps = sitk.ConnectedComponent(out_image == label)
    #     stats = _get_components_stats(comps)
    #     if len(stats) > 1:
    #         for stat in stats[1:]:
    #             print("damn")
    #             out_image -= (comps == stat[0]) * (label - to_merge_label)

    return cast(create_image(arr, main_comp))


def filtrate(segmentation: sitk.Image, th=10000, number=-1) -> sitk.Image:
    """
    Remove connected components with less than th voxels or
    leaves `number` of components
    :param segmentation: sitk image
    :param th: voxels threshold
    :param number: of components to leave
    :return: filtered image
    """
    connected = sitk.ConnectedComponent(segmentation)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(connected)

    comps_by_volume = _get_components_stats(connected)

    if number == -1:
        indexes = [x[0] for x in filter(lambda x: x[1] > th, comps_by_volume)]
    else:
        indexes = [x[0] for x in comps_by_volume][:number]

    labels = {i: 0 if i not in indexes else 1 for i in filt.GetLabels()}

    return cast(sitk.ChangeLabel(connected, labels))


def filtrate_segm(seg: sitk.Image, threshold=3000, bone_threshold=200000) -> sitk.Image:
    """
    Remove and merge connected components.
    :param seg: sitk image
    :return: filtrated segmentation
    """
    teeth = filtrate(seg == 3, th=threshold)
    upper = filtrate(seg == 1, th=350000)
    lower = sitk.ChangeLabel(seg * ((teeth + upper) == 0), {3: 0})
    lower = filtrate(lower, th=bone_threshold)

    upper = cast(upper)
    upper += cast(teeth) * 3 + cast(lower) * 2

    return cast(upper)


def __filter_outliers(centers, stats, theshold=200, volume_th=0.2):
    volume = sum(stats.values()) # total number of voxels
    center = np.array(list(centers.values())).mean(0).tolist()
    avg_dist = np.mean([l2_points_distance(v, center) for v in centers.values()]) # avg deviation
    to_drop = []
    for label in centers:
        dist = l2_points_distance(center, centers[label])
        if dist > theshold:
            cur_centers = {k: v for k, v in centers.items() if k != label}
            cur_stats = {k: v for k, v in stats.items() if k != label}
            cur_center = np.array(list(cur_centers.values())).mean(0).tolist()
            cur_avg_dist = np.mean([l2_points_distance(v, cur_center) for v in cur_centers.values()])
            if cur_avg_dist <= avg_dist and stats[label] / volume < volume_th: # not to drop significant part
                to_drop.append(label)
                to_drop.extend(__filter_outliers(cur_centers, cur_stats, theshold=theshold))

    return to_drop


def filter_outliers(image, theshold=150, volume_th=0.2, logger=logging.getLogger("CBCT")):
    components = sitk.ConnectedComponent(image == 3)
    stats = dict(_get_components_stats(components))
    centers = get_centers(components)
    volume = sum(stats.values())
    to_drop = __filter_outliers(centers, dict(stats), theshold, volume_th=volume_th)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(image)
    labels = filt.GetLabels()

    lower_jaw, lj_coords = crop_foreground(image == 2, return_coords=True) if 2 in labels else (None, None)
    lj_size = lower_jaw.GetSize() if 2 in labels else None
    upper_jaw, uj_coords = crop_foreground(image == 1, return_coords=True) if 1 in labels else (None, None)
    uj_size = upper_jaw.GetSize() if 1 in labels else None
    volume -= sum(stats[l] for l in to_drop)
    if lower_jaw is not None and uj_size is not None:
        for c in centers:
            front = max(min(lj_coords[1], uj_coords[1]) - 10, 0)
            if ((centers[c][0] < uj_coords[0] or centers[c][0] > uj_coords[0] + uj_size[0]) and \
                (centers[c][0] < lj_coords[0] or centers[c][0] > lj_coords[0] + lj_size[0])) or \
                    (centers[c][1] < front or centers[c][1] > lj_coords[1] + lj_size[1]) or \
                    (centers[c][2] < lj_coords[2] or centers[c][2] > uj_coords[2] + uj_size[2]):
                if volume != 0 and stats[c] / volume < volume_th:
                    to_drop.append(c)

    for label in to_drop:
        logger.debug(f"Postprocessing | removed outliers with volume {stats[label]}")

    return sitk.ChangeLabel(components, {k: 0 for k in to_drop}) != 0


def get_centers(seg):
    temp2 = cast(seg)
    res = {}
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(temp2)
    for label in filt.GetLabels():
        c = filt.GetCentroid(label)
        ind = temp2.TransformPhysicalPointToIndex(c)
        res[label] = ind

    return res


def l2_points_distance(a, b):
    return np.linalg.norm(np.array(a) - b)
