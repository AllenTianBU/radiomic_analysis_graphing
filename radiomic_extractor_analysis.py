import os, sys
from radiomics import featureextractor, getFeatureClasses
import SimpleITK as sitk
import radiomics
import statistics
from itertools import islice
from collections import OrderedDict
from pathlib import Path
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
import csv
import copy
from seaborn import heatmap


def randomFeatureMultipier(temp_feature_vector,a,b):
    for featurename in temp_feature_vector.keys():
        temp_feature_vector[featurename] = float(temp_feature_vector[featurename])*random.uniform(a,b)
    return temp_feature_vector
def rotate_image(scan_path, mask_path):
    scan = sitk.ReadImage(scan_path[0])
    mask = sitk.ReadImage(mask_path[0])

    scan_np = sitk.GetArrayFromImage(scan)
    pet_scan_slice = scan_np[74,:,:]
    plt.imshow(pet_scan_slice, cmap = 'grey')
    plt.xlim(100, 150)
    plt.ylim(100, 150)
    plt.title('Pet Scan (Reference)')
    plt.show()

    #Try 2D Transformation
    transform = sitk.Euler3DTransform()
    transform.SetCenter(scan.GetOrigin)
    transform.SetRotation(0,0,90)
    rotated_image = sitk.Resample(scan, scan, transform, sitk.sitkLinear, 0.0, scan.GetPixelID())

    rotated_image_np = sitk.GetArrayFromImage(rotated_image)
    pet_scan_slice = rotated_image_np[74,:,:]
    plt.imshow(pet_scan_slice, cmap = 'grey')
    plt.xlim(100, 150)
    plt.ylim(100, 150)
    plt.title('Pet Scan (Reference)')
    plt.show()
def gauss_filter(scans):
    """
    :param scans: list of scan
    :return:
    """
    filtered_scans = []
    # scan in scans
    for scan in scans:
    # for x in range(len(scans)):
        image = sitk.ReadImage(scan)
        #scan = sitk.ReadImage(scans[x])
        gaussfilter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussfilter.SetSigma(2.5)
        gaussfilter.SetNormalizeAcrossScale(True)
        filtered_scan = gaussfilter.Execute(image)

        write_path = Path(f"{scan[0:-7]}filtered.nii.gz")
        write_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(filtered_scan, write_path)
        filtered_scans.append(str(write_path))
    return filtered_scans


def file_read_folder(path):
    dirs = os.listdir(path)
    folder_list = []
    for file in dirs:
        if file != '.DS_Store':
            folder_list.append(path + file)

    # Sorting in order
    def extract_numbers(text):
        return tuple(map(int, re.findall(r'\d+', text)))
    folder_list = sorted(folder_list, key=extract_numbers)

    return folder_list


def file_read(path, start, start_mask):
    dirs = os.listdir(path)
    scans = ''
    masks = []
    for file in dirs:
        if file.startswith(start):
            scans = path +'/'+ file
        if file.startswith(start_mask):
            masks.append(path +'/'+ file)

    def extract_numbers(text):
        return tuple(map(int, re.findall(r'\d+', text)))
    masks = sorted(masks, key=extract_numbers)
    return scans, masks


def getFeatureVectorList(scan, masks, extractor):
    feature_vector = []
    for mask in masks:
        print (scan + '   ' + mask)
        scan_sitk = sitk.ReadImage(scan)
        mask_sitk = sitk.ReadImage(mask)
        resampled_mask_sitk = sitk.Resample(mask_sitk, scan_sitk, sitk.Transform()) # Linear?
        scan_origin = scan_sitk.GetOrigin()
        mask_origin = mask_sitk.GetOrigin()
        scan_direction = scan_sitk.GetDirection()
        mask_direction = mask_sitk.GetDirection()
        scan_spacing = scan_sitk.GetSpacing()
        mask_spacing = mask_sitk.GetSpacing()

        temp_feature_vector = extractor.execute(scan_sitk, resampled_mask_sitk)
        # Removing the dictionary that don't include features
        temp_feature_vector = islice(temp_feature_vector.items(), 22, 1000)
        temp_feature_vector = dict(temp_feature_vector)
        # Float conversion
        temp_feature_vector_append = {key: float(value) for key, value in temp_feature_vector.items()}
        feature_vector.append(temp_feature_vector_append)
    return feature_vector


def getFeatureCovList(feature_vector_list_of_dic):
    features_cov_list_of_dic = []
    for x in range(11): # 11
        features_cov_dic = {}# for 11 inserts
        for featureName in feature_vector_list_of_dic[0][0].keys():
            value_list = []
            for feature_vector_list in feature_vector_list_of_dic:
                value_list.append(float(feature_vector_list[x][featureName])) # Add the choosen feature value
            features_cov_dic[featureName] = cov(value_list)
        features_cov_list_of_dic.append(features_cov_dic)
    return features_cov_list_of_dic


def cov(data):
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    cov = (stdev/mean)*100
    return abs(cov)


def cov_average_func(cov_dict_list):
    cov_averaged_dict = {}
    for featureName in cov_dict_list[0].keys():
        cov_ave_value_list = []
        for cov_dict in cov_dict_list:
            cov_ave_value_list.append(cov_dict[featureName])
        cov_averaged_dict[featureName] = statistics.mean(cov_ave_value_list)
    return cov_averaged_dict


def cov_categorize(cov_averaged_dict, print_stat = False):
    cov_categorized = {}
    very_small_count = 0
    small_count = 0
    intermediate_count = 0
    large_count = 0
    for featurename in cov_averaged_dict.keys():
        cov = cov_averaged_dict[featurename]
        if cov <= 5:
            cov_categorized[featurename] = 'very small'
            very_small_count += 1
            print (featurename)
        elif cov > 5 and cov <= 10:
            cov_categorized[featurename] = 'small'
            small_count += 1
        elif cov > 10 and cov <= 20:
            print (featurename)
            cov_categorized[featurename] = 'intermediate'
            intermediate_count += 1
        else:
            cov_categorized[featurename] = 'large'
            large_count += 1

    if print_stat:
        count = str(len(cov_averaged_dict))
        #print(str(very_small_count), 'of', count, '('+ str(round(very_small_count/int(count)*100,1)) + '%)')
        #print(str(small_count), 'of', count, '(' + str(round(small_count/int(count)*100,1)) + '%)')
        #print(str(intermediate_count), 'of', count, '(' + str(round(intermediate_count/int(count)*100,1)) + '%)')
        #print(str(large_count), 'of', count, '(' + str(round(large_count/int(count)*100,1)) + '%)')

    return cov_categorized


def heatmap_plot(cov_categorize_list = None, df = None, graph_title = '', remove = ''):
    fig_size = len(cov_categorize_list[0])
    plt.figure(figsize=(20, 18.5))
    if df is None:
        df = pd.DataFrame(cov_categorize_list)
    else:
        df = df
    y_labels = ['N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11']

    #Shortening column name
    for col in df.columns:
        new_col_name = col.replace(remove, '')
        if remove == 'original_gldm_original_ngtdm_':
            new_col_name = col.replace('original_gldm_', '').replace('original_ngtdm_', '')
        df = df.rename(columns={col: new_col_name})
    heatmap = sns.heatmap(df, yticklabels= y_labels, vmax = 20, cbar_kws={'shrink': 0.6}, annot=False, cmap="OrRd", linewidths=1, linecolor='black', square = True)
    # Setting color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('COV (%)', fontsize = 40)
    colorbar.ax.tick_params(labelsize=30)
    ticks = [0, 5, 10, 15, 20]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(['0','5', '10', '15', '>20'])
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    heatmap.tick_params(axis='x', labelsize=30)
    heatmap.tick_params(axis='y', labelsize=30)
    plt.title(graph_title, fontsize=40)
    plt.tight_layout()
    #plt.savefig('/Users/allen/Desktop/Graph for presentation/' + graph_title + '.png')
    plt.show()


def exportPivotTable(feature_vector_list_of_dic, scans_list):
    stats_list = []
    for scan, feature_vector_dic in zip(scans_list, feature_vector_list_of_dic):
        #location the reconstrction time and b value
        parts = scan.split('.')
        reconstruction_time = parts[2].replace('/QClear','')
        penalized_likelihood = parts[1]

        x = 1
        for insert_dic in feature_vector_dic:
            stats_dict = {}
            stats_dict["Reconstruction Time"] = reconstruction_time
            stats_dict["Penalized-likelihood "] = penalized_likelihood
            stats_dict['Insert'] = x
            stats_dict.update(insert_dic)
            stats_list.append(stats_dict)
            x += 1
    df = pd.DataFrame(stats_list)
    for col in df.columns:
        new_col_name = col.replace('original_', '')
        df = df.rename(columns={col: new_col_name})
    df.to_csv('phantom_data.csv')


def readExcel():
    # Read excel and plot heatmap
    df = pd.read_csv('/Users/allen/PycharmProjects/pyradiomic_practice/phantom_data.csv')
    return df


def toJson(json_name = ''):
    # Read File\
    path = '/Users/allen/Desktop/P/Nifti Phantom Data/'
    folder_list_path = file_read_folder(path)
    scans_list = []
    masks_list = []
    for folder in folder_list_path:
        scans_qclear, masks_qclear = file_read(folder, 'QClear', 'ROI')
        scans_list.append(scans_qclear)
        masks_list.append(masks_qclear)

    # Class and extraction set
    featureClasses = getFeatureClasses()
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')

    # Calculating features for each file and saving in feature_vector_qclear list, as well as removing the dictionary without the features
    feature_vector_list_of_dic = []
    for scan, masks in zip(scans_list, masks_list):
        feature_vector = getFeatureVectorList(scan, masks, extractor)
        feature_vector_list_of_dic.append(feature_vector)

    with open(json_name, 'w') as file:
        json.dump(feature_vector_list_of_dic, file)


def featureSelect(feature_vector_list_of_dic, chosen_features = '', exclude_first = False, exclude_second = False, exclude_last = False):
    temp_feature_vector_list_of_dic = copy.deepcopy(feature_vector_list_of_dic)
    if exclude_second:
        del temp_feature_vector_list_of_dic[0]
        del temp_feature_vector_list_of_dic[0]
    elif exclude_first:
        del temp_feature_vector_list_of_dic[0]
    if exclude_last:
        del temp_feature_vector_list_of_dic[-1]

    if chosen_features == "original_gldm_original_ngtdm_":
        chosen_features = ['original_gldm_', 'original_ngtdm_']
        for feature_vector_list in temp_feature_vector_list_of_dic:
            for feature_vector in feature_vector_list:
                keys_to_remove = [key for key in feature_vector if chosen_features[0] not in key and chosen_features[1] not in key]
                for key in keys_to_remove:
                    del feature_vector[key]
    else:
        for feature_vector_list in temp_feature_vector_list_of_dic:
            for feature_vector in feature_vector_list:
                keys_to_remove = [key for key in feature_vector if chosen_features not in key]
                for key in keys_to_remove:
                    del feature_vector[key]
    return temp_feature_vector_list_of_dic


if __name__ == '__main__':
    # Graph needed
    graph_name_20min = [
        "First Order Features",
        "Shape Features",
        "GLCM Features",
        "GLRLM Features",
        "GLSZM Features",
        "GLDM & NGTDM Features",
    ]

    key_word_list = [
        "original_firstorder_",
        "original_shape_",
        "original_glcm_",
        "original_glrlm_",
        "original_glszm_",
        "original_gldm_original_ngtdm_",
    ]

    graph_name_600b = [
        "First Order Features",
        "Shape Features",
        "GLCM Features",
        "GLRLM Features",
        "GLSZM Features",
        "GLDM & NGTDM Features",
    ]

    json_write_read_name = 'all features (600 b)'

    #toJson(json_name = json_write_read_name)

    with open(json_write_read_name, 'r') as file:
        feature_vector_list_of_dic = json.load(file)

    for graph_name, key_word in zip(graph_name_600b, key_word_list):
        temp_feature_vector_list_of_dic = featureSelect(feature_vector_list_of_dic,
                                                        key_word,
                                                        exclude_first = True,
                                                        exclude_second = False,
                                                        exclude_last = False)
        features_cov_list_of_dict = getFeatureCovList(temp_feature_vector_list_of_dic)
        cov_average = cov_average_func(features_cov_list_of_dict)

        #print(graph_name)
        cov_categorize(cov_average, print_stat = True)

        graph_name = graph_name #+ ' 200-800β'
        heatmap_plot(cov_categorize_list = features_cov_list_of_dict,
                     graph_title = graph_name,
                     remove = key_word)

    print("End")
