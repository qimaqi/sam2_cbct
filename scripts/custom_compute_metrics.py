import os
import nibabel as nib
 
import numpy as np
import SimpleITK as sitk
import numpy as np
import scipy
import json
from pathlib import Path
import os
from typing import Dict

 
from evalutils.io import SimpleITKLoader
import pandas as pd
import argparse
import glob
from tqdm import tqdm
import shutil

from scipy.ndimage import label
 
from cbct_utils import common
from scipy.spatial import cKDTree
from medpy.metric import binary


# from align_label import labels as LABELS
LABELS =   {
        "background": 0,
        "spleen": 1,
        "right kidney": 2,
        "left kidney": 3,
        "gall bladder": 4,
        "esophagus": 5,
        "liver": 6,
        "stomach": 7,
        "arota": 8,
        "postcava": 9,
        "pancreas": 10,
        "right adrenal gland": 11,
        "left adrenal gland": 12,
        "duodenum": 13,
        "bladder": 14,
        "prostate/uterus": 15
    }



import pathlib
import k3d
 
from cbct_utils.common import resample, to_numpy


def compute_binary_dice(pred, label):
    addition = pred.sum() + label.sum()
    if addition == 0:
        return 1.0
    return 2. * np.logical_and(pred, label).sum() / addition


def mean(l):
    if len(l) == 0:
        return 0
    return sum(l)/len(l)

def compute_binary_hd95(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return np.linalg.norm(pred.shape)
    return binary.hd95(pred, gt)

SLICER_COLOR_MAP = [8433280, 15849105, 11631205, 7321810, 14181711, 14516837, 9498256, 12609624, 14480660, 320496,
                    16775900, 15129670, 13158635, 16448210, 16045617, 38862, 14181711, 12033244, 12048083, 10010063,
                    7321810, 11719922, 4500580, 7325059, 5618943, 37150, 14083714, 320496, 14352383, 11205370, 9232612,
                    12337436, 14204888, 9518146, 9855571, 11631205, 16045617] + [2353] * 167 + [12221601, 11631205,
                                                                                                16777180, 15395522,
                                                                                                13405874, 11827097,
                                                                                                14189673, 16776677,
                                                                                                13477774, 13412495,
                                                                                                16769223, 14516837,
                                                                                                37150, 9148002,
                                                                                                16364655, 10316962,
                                                                                                13338740, 12150355,
                                                                                                12150355, 16234148,
                                                                                                16234148, 14588548,
                                                                                                8174303, 16366230,
                                                                                                16366230, 16034451,
                                                                                                16758174, 16760485,
                                                                                                14915970, 13995377,
                                                                                                13995377, 12680039]
 
 
def generate_html_model(image: sitk.Image, out_path: pathlib.Path,
                        spacing: tuple = (0.6, 0.6, 0.6),
                        compression_level=3):
    image = resample(image, spacing)
    lower = to_numpy(image == 2) * 2
    upper = to_numpy(image == 1)
    teeth = to_numpy(image - (image == 1) - (image == 2) * 2)
    try:
        plot = k3d.plot(name='points', grid_visible=False)
        plot += k3d.voxels(teeth.transpose(2, 1, 0), color_map=SLICER_COLOR_MAP, compression_level=compression_level) + \
                k3d.voxels(upper.transpose(2, 1, 0), color_map=SLICER_COLOR_MAP, compression_level=compression_level) + \
                k3d.voxels(lower.transpose(2, 1, 0), color_map=SLICER_COLOR_MAP, compression_level=compression_level)
 
        _ = plot.display()
 
        with open(out_path, 'w+') as fp:
            fp.write(plot.get_snapshot())
    except:
        pass
 
 
 
def compute_multiclass_dice_and_hd95(pred, mask):
 
    dice_per_class = {}
    hd_per_class = {}
 
    for label_name, label_id in LABELS.items():
        # label_id = label.id
        # label_name = label.name
        # label_id = LABELS[label]

        binary_class_pred = pred == label_id
        #  print("binary_class_pred", type(binary_class_pred), binary_class_pred.sum())
        binary_class_label = mask == label_id
        dice = compute_binary_dice(binary_class_pred, binary_class_label)
        hd = compute_binary_hd95(binary_class_pred, binary_class_label)
        dice_per_class[label_name] = dice
        hd_per_class[label_name] = hd
        # print("label_name", label_name, label_id, dice)
   
    # when calculate average, we do not want consider value = 1 for dice and value = 0 for hd
 
    dice_per_class['average'] = mean(dice_per_class.values())
    hd_per_class['average'] = mean(hd_per_class.values())
    return dice_per_class, hd_per_class
 
 
 
if __name__ == "__main__":
    """
    data preprocess need to calculate the basic information of data, including:
    1) the volume size of each slices
    2) the intensity of the volume
    3) generate a train, label split for further use
 
    """
    parser = argparse.ArgumentParser(description="preprocess data function in total (CT data)")
    parser.add_argument("--eval_dir", type=str, help="CT Volume root directory", default='/hdd1/qimaqi/nrrd_dataset/conveted_nii/align_baseline/finish_result/')
    parser.add_argument("--mask_dir", type=str, default='/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_freeze_pred_cbct_gt', help="org seg results directory")
    parser.add_argument("--gt_suffix", type=str, required=False, default=None,
                        help="Suffix of gt input files")
    args = parser.parse_args()
 
    # get all the volume paths
    eval_dir = args.eval_dir
 
 
    volume_paths = glob.glob(os.path.join(eval_dir, "*.nii.gz"))
    volume_paths = sorted(volume_paths)
    loader = SimpleITKLoader()
 
    results = {
            'eval_dir': eval_dir,
            'eval_results': [],
            }
   
    result_save_path = os.path.join(eval_dir, f"eval_results.json")
 
    try:
        with open(result_save_path, "r") as f:
            results = json.load(f)
            valid_num = len(results['eval_results'])
    except:
        valid_num = 0
 
    if not os.path.exists(result_save_path) or valid_num != len(volume_paths):
        for volume_i in tqdm(volume_paths):
            # check volume name if in results or not
            volume_name = volume_i.split("/")[-1].split(".nii.gz")[0]
            print("volume_name", volume_name)
            if len(results['eval_results']) > 0:
                # print("check volume_name", volume_name)
                if volume_name in [i['name'] for i in results['eval_results']]:
                    print(f"skip {volume_name}")
                    continue
 
            results_i = {'outlier_pred': [], 'small_pred':[]}
            # {
            #     'DiceCoefficient': [],
            #     'HausdorffDistance95': [],
            #     'name': []}
     
 
            results_i['name'] =volume_name
            pred_path = volume_i
            # seg_name = volume_name.replace("cbct_segmentation", f"{args.gt_suffix}.nrrd")
            # print("volume_name", volume_name)
            # print("seg_name", seg_name)
            if args.gt_suffix is None:
                seg_name = volume_name + '.nii.gz'
            else:
                seg_name = volume_name.replace("cbct_segmentation", f"{args.gt_suffix}.nii.gz")
            seg_path = os.path.join(args.mask_dir, seg_name)
            # print("seg_path", seg_path)
            pred = loader.load_image(pred_path)
            gt = loader.load_image(seg_path)
            # to 8-bit  integer
            pred = sitk.Cast(pred, sitk.sitkUInt8)
            gt = sitk.Cast(gt, sitk.sitkUInt8)
            # go over and ignore label if number <30

            generate_html_model(pred, os.path.join(eval_dir, f"{volume_name}_pred.html"))
            generate_html_model(gt, os.path.join(eval_dir, f"{volume_name}_gt.html"))
            # pred = sitk.Cast(pred,2)
            # gt = sitk.Cast(gt,2)
     
 
            pred = sitk.GetArrayFromImage(pred).squeeze()
            gt = sitk.GetArrayFromImage(gt).squeeze()
 
            dice, hd95 = compute_multiclass_dice_and_hd95(pred, gt)
 
            print("==================")
            print("pred_name", pred_path)
            print("dice", dice)
            print("hd95", hd95)
 
 
            results_i['DiceCoefficient'] = dice
            results_i['HausdorffDistance95'] = hd95
            results_i['name'] =volume_name
            results['eval_results'].append(results_i)
 
            with open(result_save_path, "w") as f:
                json.dump(results, f, indent=4)
 
        # save the results
        with open(result_save_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        with open(result_save_path, "r") as f:
            results = json.load(f)
    # print mean metric
    # print("results", results)
    # df = pd.DataFrame(results['eval_results'])
    dice_list = []
    hd95_list = []
 
    summerize_dict = {}
    # for label_i in gt_labels[1:]:
    #     category = label_i.category
    #     label_name = label_i.name
    #     if args.implant:
    #         if category != 'others':
    #             continue
    #     if args.old:
    #         # change label_id to the old label_id
    #         label_name.replace("UR ", "Upper Right ")
    #         label_name.replace("UL ", "Upper Left ")
    #         label_name.replace("LR ", "Lower Right ")
    #         label_name.replace("LL ", "Lower Left ")
 
    #     summerize_dict[label_name] = []
 
 
    for i in results['eval_results']:
        # print("i", i['DiceCoefficient'].keys())
        dice_list.append(float(i['DiceCoefficient']['average']))
        hd95_list.append(float(i['HausdorffDistance95']['average']))
        # for label_i in gt_labels[1:]:
        #     label_name = label_i.name
        #     category = label_i.category
        for label_name, label_id in LABELS.items():
            # if args.implant:
            #     if category != 'others':
            #         continue
            # if args.old:
            #     # change label_id to the old label_id
            #     label_name = label_name.replace("UR", "Upper Right")
            #     label_name = label_name.replace("UL", "Upper Left")
            #     label_name = label_name.replace("LR", "Lower Right")
            #     label_name = label_name.replace("LL", "Lower Left")
 
            # print("label_name", label_name)
            # summerize_dict[label_name].append(i['DiceCoefficient'][label_name])
            if i['DiceCoefficient'][label_name] != 1:
                summerize_dict[label_name].append(i['DiceCoefficient'][label_name])
        # clDice_list.extend(i['clDice'])
 
    print("DiceCoefficient", np.mean(dice_list))
    print("HausdorffDistance95", np.mean(hd95_list))
 
    for label_name, label_id in LABELS.items():
 
    # for key in gt_labels[1:]:
    #     category = key.category
    #     label_name = key.name
 
        # if args.implant:
        #     if category != 'others':
        #         continue
        summerize_dict[label_name] = np.mean(summerize_dict[label_name])
    #
    with open(os.path.join(eval_dir, f"org_size_{use_org_size}_summerize_dict.json"), "w") as f:
        json.dump(summerize_dict, f, indent=4)
 
 