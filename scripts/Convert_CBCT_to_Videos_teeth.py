from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage
import shutil
# import multiprocessing
from multiprocessing import Pool
from functools import partial



def CTNormalizationTeeth(image: np.ndarray, seg: np.ndarray = None, to_uint8=False) -> np.ndarray:
    mean_intensity =893.993896484375
    std_intensity = 770.8853149414062
    lower_bound = -493.0
    upper_bound = 3835.0
    image = image.astype(np.float32, copy=False)
    np.clip(image, lower_bound, upper_bound, out=image)
    # then normalize the image

    if to_uint8:
        # using lower bound and upper bound to normalize the image to 0-255
        image -= lower_bound
        image /= (upper_bound - lower_bound)
        image *= 255
        image = image.astype(np.uint8)
        # new mean = mean_intensity - lower_bound / (upper_bound - lower_bound) * 255 = 0.320
        # new std = std_intensity / (upper_bound - lower_bound) * 255 = 0.291
    else:
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        
    return image



def crop_pad_to_target_size(image,label,target_size):
    z, h, w = image.shape
    assert label.shape == image.shape
    # first we treat h dimension
    if h < target_size[0]:
        # doing padding on h dimension
        pad_h = (target_size[0] - h) // 2
        pad_h_total = target_size[0] - h
        if pad_h_total % 2 == 0:
            pad_h = (pad_h, pad_h)
        else:
            pad_h = (pad_h, pad_h + 1)
    else:
        # doing cropping on h dimension
        pad_h = (0, 0)
        pad_h_total = 0
    # then we treat w dimension
    if w < target_size[1]:
        # doing padding on w dimension
        pad_w = (target_size[1] - w) // 2
        pad_w_total = target_size[1] - w
        if pad_w_total % 2 == 0:
            pad_w = (pad_w, pad_w)
        else:
            pad_w = (pad_w, pad_w + 1)
    else:
        # doing cropping on w dimension
        pad_w = (0, 0)
        pad_w_total = 0
    
    # we can ignroe z dimension
    pad_z = (0, 0)
    pad_z_total = 0
    # now we do the actual padding
    pad = (pad_z, pad_h, pad_w)
    image = np.pad(image, pad, mode='constant', constant_values=0)
    label = np.pad(label, pad, mode='constant', constant_values=0)

    # after padding we check if we need to crop or not
    z, h, w = image.shape
    if h > target_size[0]:
        start_h = (h - target_size[0]) // 2
        end_h = start_h + target_size[0]
    else:
        start_h = 0
        end_h = h
    if w > target_size[1]:
        start_w = (w - target_size[1]) // 2
        end_w = start_w + target_size[1]
    else:
        start_w = 0
        end_w = w
    image = image[:, start_h:end_h, start_w:end_w]
    label = label[:, start_h:end_h, start_w:end_w]

    return image, label

def normalize_hist(image: sitk.Image, th=0.999) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(image)
    ns, intensity = np.histogram(arr.reshape(-1), bins=256)

    cutoff_1 = np.ediff1d(ns[:-1]).argmax(axis=0) + 1
    total = np.sum(ns[cutoff_1 + 1:-1])
    cutoff_2 = (np.cumsum(ns[cutoff_1 + 1:].astype(int) / total) > th).argmax() + cutoff_1
    image = sitk.Clamp(image, outputPixelType=sitk.sitkFloat32,
                       lowerBound=float(intensity[cutoff_1]), upperBound=float(intensity[cutoff_2]))

    return sitk.RescaleIntensity(image)


def volume_to_video(volume, save_path, data_tpye='image'):
    os.makedirs(save_path, exist_ok=True)
    if data_tpye == 'image':
        ext = '.jpg'
    elif data_tpye == 'label':
        ext = '.png'
    elif data_tpye == 'label_vis':
        ext = '.png'
    depth = volume.shape[0]
    for frame_i in range(depth):
        frame = volume[frame_i]
        # how to map to 255? using align historgam normalization
        # frame = frame.astype(np.int8)
        frame_path = join(save_path, f'{frame_i:08d}{ext}')
        # saving image using PIL 
        if data_tpye == 'label':
            frame = frame.astype(np.int32)
            # ignore all labels >= 56:
            frame[frame >= 36] = 0
            frame = PILImage.fromarray(frame)
            # convert to  color-pattlate
            # print("frame before pattlate", np.shape(frame), np.min(frame), np.max(frame))
            frame = frame.convert('P')
            # print("frame", np.shape(frame), np.min(frame), np.max(frame))
            # frame.putpalette(PILImage.ADAPTIVE)
            # print("frame adapt palette", np.shape(frame), np.min(frame), np.max(frame))
            # frame = frame.convert('P')
            # print("frame back to palette", np.shape(frame), np.min(frame), np.max(frame))
            # print("frame", np.shape(frame), np.min(frame), np.max(frame))
        elif data_tpye == 'label_vis':
            frame = frame.astype(np.int32)
            frame[frame >= 36] = 0
            # using the color map
            from matplotlib import cm
            # using color map to show different teeth labels
            cmap = cm.get_cmap('tab20')
            frame = cmap(frame)
            frame = (frame * 255).astype(np.uint8)
            frame = PILImage.fromarray(frame)
        else:
            # for a normalized image, how to convert to 0-255
            frame = PILImage.fromarray(frame).convert('RGB')

        frame.save(frame_path)
    # compress the video folder using shutil
    shutil.make_archive(save_path, 'zip', save_path)
    # remove the folder
    shutil.rmtree(save_path)


def convert_main_func(image_file, images_dir, labels_dir, save_root_dir, trian_files):

    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, image_file).replace('_0000.nii.gz', '.nii.gz')
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    # image = normalize_hist(image)
    image = sitk.GetArrayFromImage(image)
    label = sitk.GetArrayFromImage(label)
    image = CTNormalizationTeeth(image, to_uint8=True)

    # print("before crop and pad", image.shape, label.shape)
    image_, label_ = crop_pad_to_target_size(image, label, [512, 512])

    image_file_name = image_file.replace('_0000.nii.gz', '')
    if len(trian_files) > 0:
        if image_file_name in trian_files:
            split_i = 'train'
        else:
            split_i = 'val'
    else:
        split_i = 'train'

    image_save_path = os.path.join(save_root_dir, split_i, 'JPEGImages', image_file_name)
    volume_to_video(image_, image_save_path, data_tpye='image')
    label_save_path = os.path.join(save_root_dir, split_i, 'Annotations', image_file_name)
    volume_to_video(label_, label_save_path, data_tpye='label')
    label_vis_save_path = os.path.join(save_root_dir, split_i, 'Annotations_vis', image_file_name)
    volume_to_video(label_, label_vis_save_path, data_tpye='label_vis')




if __name__ == "__main__":
    input_root_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/'
    save_root_dir = '/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/teeth_datasets/teeth_videos_new'
    images_dir = os.path.join(input_root_dir, 'imagesTr')
    labels_dir = os.path.join(input_root_dir, 'labelsTr')
    splits_file = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/splits_final.json'


    # os.makedirs(save_root_dir, exist_ok=True)
    images_files = os.listdir(images_dir)
    images_files = sorted(images_files)

    with open(splits_file, 'r') as f:
        splits = json.load(f)[0]
    trian_files = splits['train']
    test_files = splits['val']

    func = partial(
        convert_main_func, 
        images_dir=images_dir, 
        labels_dir=labels_dir, 
        save_root_dir=save_root_dir, 
        trian_files=trian_files
    )

    with Pool(4) as p:
        # func = partial(convert_main_func, images_dir=images_dir, labels_dir=labels_dir, save_root_dir=save_root_dir, trian_files=trian_files)
        # p.map(func, images_files)
        # add tqdm to show the progress in multiprocessing
        results = list(tqdm(p.imap(func, images_files), total=len(images_files)))




    # for step, image_file in tqdm(enumerate( images_files), total=len(images_files)):
    #     image_path = os.path.join(images_dir, image_file)
    #     label_path = os.path.join(labels_dir, image_file).replace('_0000.nii.gz', '.nii.gz')
    #     image = sitk.ReadImage(image_path)
    #     label = sitk.ReadImage(label_path)
    #     # image = normalize_hist(image)
    #     image = sitk.GetArrayFromImage(image)
    #     label = sitk.GetArrayFromImage(label)
    #     image = CTNormalizationTeeth(image, to_uint8=True)
    
    #     # print("before crop and pad", image.shape, label.shape)
    #     image_, label_ = crop_pad_to_target_size(image, label, [512, 512])

    #     image_file_name = image_file.replace('_0000.nii.gz', '')
    #     if len(trian_files) > 0:
    #         if image_file_name in trian_files:
    #             split_i = 'train'
    #         else:
    #             split_i = 'val'
    #     else:
    #         split_i = 'train'

    #     image_save_path = os.path.join(save_root_dir, split_i, 'JPEGImages', image_file_name)
    #     volume_to_video(image_, image_save_path, data_tpye='image')
    #     label_save_path = os.path.join(save_root_dir, split_i, 'Annotations', image_file_name)
    #     volume_to_video(label_, label_save_path, data_tpye='label')
    #     label_vis_save_path = os.path.join(save_root_dir, split_i, 'Annotations_vis', image_file_name)
    #     volume_to_video(label_, label_vis_save_path, data_tpye='label_vis')

        # print("after crop and pad", image_.shape, label_.shape)

        # break
        