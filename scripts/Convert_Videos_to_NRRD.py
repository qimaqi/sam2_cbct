from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import nibabel as nib
# import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage
import shutil
import SimpleITK as sitk

def video_to_volume(volume, save_path):
    print("saving volume to", save_path)
    # nib_volume = nib.Nifti1Image(volume, np.eye(4))
    # nib.save(nib_volume, save_path)
    # header = {
    #     'type': 'uint8',
    #     'dimension': 3,
    #     'space': 'left-posterior-superior',
    #     'sizes': volume.shape,
    #     'space directions': np.eye(3),
    #     'space origin': np.zeros(3),
    # }
    # nrrd.write(save_path, volume)
    nrrd_volume = sitk.GetImageFromArray(volume)
    # add spacing information
    nrrd_volume.SetSpacing([0.3, 0.3, 0.3])
    nrrd_volume = sitk.Cast(nrrd_volume, sitk.sitkUInt8)
    # set origin to 0
    nrrd_volume.SetOrigin([0, 0, 0])
    # set direction to (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    nrrd_volume.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    sitk.WriteImage(nrrd_volume, save_path)




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


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_freeze_pred')
    # parser.add_argument('--save_root_dir', type=str, default='/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_freeze_pred_cbct_gt')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    input_root_dir = args.input_path
    input_name = input_root_dir.split('/')[-1]
    save_root_dir = os.path.join('/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/', input_name) + '_cbct'
    os.makedirs(save_root_dir, exist_ok=True)

    # input_root_dir = '/cluster/work/cvl/qimaqi/miccai_2025/Dataset219_AMOS2022_videos/train/Annotations' #'/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_freeze_pred/'
    # save_root_dir = '/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_freeze_pred_cbct_gt'
    # splits_file = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/splits_final.json'
    # os.makedirs(save_root_dir, exist_ok=True)
    video_dirs = os.listdir(input_root_dir)
    video_dirs = [video_dir_i for video_dir_i in video_dirs if not video_dir_i.endswith('.zip')]

    for video_dir_i in tqdm(video_dirs):
        video_dir_i_path = os.path.join(input_root_dir, video_dir_i)
        frames_i = os.listdir(video_dir_i_path)
        frames_i = sorted(frames_i)
        frames_i = [frame_i for frame_i in frames_i if frame_i.endswith('.png') or frame_i.endswith('.jpg')]
        Depth = len(frames_i)
        Height, Width = PILImage.open(os.path.join(video_dir_i_path, frames_i[0])).size
     
        volume = np.zeros((Depth, Height, Width), dtype=np.float32)
        print("volume shape", volume.shape)
        for frame_i in range(Depth):
            frame = PILImage.open(os.path.join(video_dir_i_path, frames_i[frame_i]))
            frame = np.array(frame)
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=-1)
                # print("frame shape", frame.shape, np.min(frame), np.max(frame), np.mean(frame), np.nonzero(frame))
                volume[frame_i] = frame
            else:
                volume[frame_i] = frame

        save_path = os.path.join(save_root_dir, video_dir_i + 'cbct_segmentation.nrrd')
        # convert np to nibabel
        video_to_volume(volume, save_path)

    # for step, image_file in tqdm(enumerate( images_files), total=len(images_files)):
    #     image_path = os.path.join(images_dir, image_file)
    #     label_path = os.path.join(labels_dir, image_file).replace('_0000.nii.gz', '.nii.gz')
    #     image = sitk.ReadImage(image_path)
    #     label = sitk.ReadImage(label_path)
    #     image = normalize_hist(image)
    #     image = sitk.GetArrayFromImage(image)
    #     label = sitk.GetArrayFromImage(label)
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

        # print("after crop and pad", image_.shape, label_.shape)

        # break
        