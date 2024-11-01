import numpy as np
import os 

raw_files_list = '/cluster/work/cvl/qimaqi/miccai_2025/Dataset219_AMOS2022_videos/train/JPEGImages'
raw_files = os.listdir(raw_files_list)
raw_files = sorted(raw_files)

print(raw_files, len(raw_files))

for fold_num in range(5):
    eval_fold = []
    train_fold = []
    # using seed to make sure the split is the same for each run
    np.random.seed(42)
    # shuffle the list
    np.random.shuffle(raw_files)
    # split the list into 5 folds
    fold_size = len(raw_files) // 5

    if fold_num == 4:
        eval_fold.extend(raw_files[fold_num*fold_size:])
        train_fold.extend(raw_files[:fold_num*fold_size])
    else:
        eval_fold.extend(raw_files[fold_num*fold_size:(fold_num+1)*fold_size])
        train_fold.extend(raw_files[:fold_num*fold_size])
        train_fold.extend(raw_files[(fold_num+1)*fold_size:])

    # save to txt file, each file take one line

    np.savetxt(f'/cluster/work/cvl/qimaqi/miccai_2025/Dataset219_AMOS2022_videos/train/eval_fold_{fold_num}.txt', eval_fold, fmt='%s')
    np.savetxt(f'/cluster/work/cvl/qimaqi/miccai_2025/Dataset219_AMOS2022_videos/train/train_fold_{fold_num}.txt', train_fold, fmt='%s')

    # with open(f'/cluster/work/cvl/qimaqi/miccai_2025/Dataset219_AMOS2022_videos/train/fold_{fold_num}.txt', 'w') as f:
    #     for file_name in fold:
    #         f.write(f"{file_name}\n")
            