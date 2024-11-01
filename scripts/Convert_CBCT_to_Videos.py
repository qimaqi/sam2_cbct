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
    depth = volume.shape[0]
    for frame_i in range(depth):
        frame = volume[frame_i]
        # how to map to 255? using align historgam normalization
        # frame = frame.astype(np.int8)
        frame_path = join(save_path, f'{frame_i:08d}{ext}')
        # saving image using PIL 
        if data_tpye == 'label':
            frame = frame.astype(np.int8)
            # ignore all labels >= 56:
            frame[frame >= 56] = 0
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
        else:
            frame = PILImage.fromarray(frame).convert('RGB')

        frame.save(frame_path)
    # compress the video folder using shutil
    shutil.make_archive(save_path, 'zip', save_path)
    # remove the folder
    shutil.rmtree(save_path)

    #


if __name__ == "__main__":
    input_root_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/'
    save_root_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/videodata/Dataset219_AMOS2022_videos'
    images_dir = os.path.join(input_root_dir, 'imagesTr')
    labels_dir = os.path.join(input_root_dir, 'labelsTr')
    # splits_file = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/splits_final.json'


    # os.makedirs(save_root_dir, exist_ok=True)
    images_files = os.listdir(images_dir)
    images_files = sorted(images_files)

    # with open(splits_file, 'r') as f:
    #     splits = json.load(f)[0]
    # trian_files = splits['train']
    # test_files = splits['val']
    trian_files = []
    
    for step, image_file in tqdm(enumerate( images_files), total=len(images_files)):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file).replace('_0000.nii.gz', '.nii.gz')
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image = normalize_hist(image)
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
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

        # print("after crop and pad", image_.shape, label_.shape)

        # break
        