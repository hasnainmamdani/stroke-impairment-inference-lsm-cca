from nilearn.image import load_img, resample_to_img
from nilearn.datasets import load_mni152_brain_mask
import os
import glob
import numpy as np
import pandas as pd
np.random.seed(39)


def get_mni_mask(reference_img):

    mask_img = load_mni152_brain_mask()
    mask_img_resampled = resample_to_img(mask_img, reference_img, interpolation="linear")
    mask = np.where(mask_img_resampled.get_fdata() == 1)  # Not using NiftiMasker because it takes too long and too much memory to transform.

    return mask


def get_binary_lesion_imgs(data_dir):

    if not os.path.isfile(data_dir + "binary_imgs.npy"):

        dataset_path = data_dir + "stroke-dataset/HallymBundang_lesionmaps_n1432_24032020/"
        img_filenames = glob.glob(os.path.join(dataset_path, "*.nii.gz"))
        img_filenames.sort()
        print("Number of subjects: %d" % len(img_filenames))

        mask = get_mni_mask(load_img(img_filenames[0]))

        print(mask[0].shape)
        img_data = np.empty((len(img_filenames), len(mask[0])), dtype=bool)

        for i in range(len(img_filenames)):
            print(i)
            img_data[i] = load_img(img_filenames[i]).get_fdata()[mask].astype(bool)

        np.save(data_dir + "binary_imgs", img_data)

    else:
        img_data = np.load(data_dir + "binary_imgs.npy")

    return img_data


def load_cort_subcort_cereb_combined_atlas(atlas_dir, reference_img, mask):

    print("loading Harvard Oxford Cortical and Subcortical atlas and Cerebellum regions of the hammer's atlas...")

    atlas_gm_img = load_img(atlas_dir + "HarvardOxford_cerebellum_combined_01102020.nii")
    labels_gm = np.array(pd.read_excel(atlas_dir + "HarvardOxford_cerebellum_combined_01102020.xlsx")["Structure"], dtype="U")
    atlas_gm_img_resampled = resample_to_img(atlas_gm_img, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_gm_vectorized = atlas_gm_img_resampled.get_fdata()[mask].astype(int)

    print(len(labels_gm))
    print(atlas_gm_img_resampled.shape)
    print(atlas_gm_vectorized.shape)
    print(np.unique(atlas_gm_vectorized))

    return atlas_gm_vectorized, labels_gm


def load_jhu_wm_atlas(atlas_dir, reference_img, mask):

    print("loading JHU White Matter Tract Atlas...")
    atlas_wm_img = load_img(atlas_dir + "rICBM_DTI_81_WMPM_FMRIB58.nii")
    labels_wm = np.array(pd.read_excel(atlas_dir + "ICBM__DTI81_01102020.xlsx")["Structure"], dtype="U")
    labels_wm = np.core.defchararray.add("WM - ", labels_wm)

    atlas_wm_img_resampled = resample_to_img(atlas_wm_img, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_wm_vectorized = atlas_wm_img_resampled.get_fdata()[mask].astype(int)

    print(len(labels_wm))
    print(atlas_wm_img_resampled.shape)
    print(atlas_wm_vectorized.shape)
    print(np.unique(atlas_wm_vectorized))

    return atlas_wm_vectorized, labels_wm


def create_lesion_load_matrix_atlas(atlas, region_names, imgs):

    assert atlas.shape == imgs[0].shape, "Atlas dimension and image dimension must be same"

    region_labels = np.unique(atlas)
    region_labels = region_labels[region_labels != 0]  # background/unlabeled

    lesion_load_matrix = np.zeros((imgs.shape[0], len(region_labels)), dtype=int)

    for i_label in range(len(region_labels)):

        idx = np.where(atlas == region_labels[i_label])[0]

        imgs_region = imgs[:, idx]

        lesion_load_matrix[:, i_label] = np.sum(imgs_region, axis=1)

    return lesion_load_matrix, region_names[region_labels]


def create_lesion_load_matrix(data_dir, lesion_data, reference_img, save_interm_labels=True):

    mask = get_mni_mask(reference_img)
    atlas_dir = data_dir + "atlas/by_nick_01102020/"

    atlas_gm_vectorized, atlas_gm_labels = load_cort_subcort_cereb_combined_atlas(atlas_dir, reference_img, mask)
    atlas_wm_vectorized, atlas_wm_labels = load_jhu_wm_atlas(atlas_dir, reference_img, mask)

    if save_interm_labels:

        np.save(atlas_dir + "processed/" + "ho_cort_subcort_cereb_combined_atlas_vectorized.npy", atlas_gm_vectorized)
        np.save(atlas_dir + "processed/" + "ho_cort_subcort_cereb_combined_atlas_region_labels.npy", atlas_gm_labels)

        np.save(atlas_dir + "processed/" + "jhu_wm_atlas_vectorized.npy", atlas_wm_vectorized)
        np.save(atlas_dir + "processed/" + "jhu_wm_atlas_region_labels.npy", atlas_wm_labels)

    llm_gm, region_names_gm = create_lesion_load_matrix_atlas(atlas_gm_vectorized, atlas_gm_labels, lesion_data)
    llm_wm, region_names_wm = create_lesion_load_matrix_atlas(atlas_wm_vectorized, atlas_wm_labels, lesion_data)

    llm = np.concatenate([llm_gm, llm_wm], axis=1)
    region_names = np.concatenate([region_names_gm, region_names_wm])

    return llm, region_names


def main():

    DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"

    lesion_imgs = get_binary_lesion_imgs(DATA_DIR)
    print("lesion data loaded:", lesion_imgs.shape)

    reference_img = load_img(DATA_DIR + "stroke-dataset/HallymBundang_lesionmaps_n1432_24032020/1001.nii.gz")

    lesion_load_matrix, region_labels = create_lesion_load_matrix(DATA_DIR, lesion_imgs, reference_img, True)

    print(lesion_load_matrix.shape, region_labels.shape)
    np.save(DATA_DIR + "llm/cca/combined_lesions_load_matrix_05102020.npy", lesion_load_matrix)
    np.save(DATA_DIR + "llm/cca/combined_atlas_region_labels_05102020.npy", region_labels)


if __name__ == "__main__":
    main()


