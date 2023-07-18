"""
VINCI Construction

Author: Iago Martinelli Lopes (iago.martinelli-lopes@vinci-construction.com)
Tutor: Djamil Yahia-Ouahmed
Date: 11/06/2020
Model : Unidecode
Objective : 1) Fix some problems with json files like "*.json0"
            2) Remove accents, special characters and spaces from file name
            3) Verify that all JSON files have an associated image
"""

import os
import shutil
import math
import json
import random
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from os import path
import cv2
from sklearn.model_selection import train_test_split
import unidecode
import posixpath

ImageFile.LOAD_TRUNCATED_IMAGES = True


# MAIN function with INPUTS and INSTRUCTIONS is located in the end of this file


def iterate_folder_test(root_dir: str, s) -> None:
    """
    :param root_dir:
    :param s:
    :return:
    """
    folder = os.listdir(root_dir)
    dirs = [os.path.join(root_dir, f) for f in folder if os.path.isdir(os.path.join(root_dir, f))]
    dirs_smallpath = [f for f in folder if os.path.isdir(os.path.join(root_dir, f))]

    if 'ann' in dirs_smallpath and 'img' in dirs_smallpath and len(dirs_smallpath) != 2:
        raise Exception("Expected 'ann' and 'img' folders only, but more than one was found."
                        "Folder: {}".format(root_dir))

    elif 'ann' in dirs_smallpath and 'img' in dirs_smallpath:
        dirs = sorted(dirs)
        path_ann = dirs[0]
        files = list(sorted(os.listdir(path_ann)))
        for f in tqdm(files):
            f_path = os.path.join(path_ann, f)
            with open(f_path) as json_file:
                data = json.load(json_file)
                json_file.close()
                objs = data["objects"]
                for obj in objs:
                    s.add(obj["classTitle"])
    else:
        for dir in dirs:
            iterate_folder_test(dir, s)


def iterate_folder(root_dir: str,polygon,list_dir,class_list) -> None:
    """
    :param root_dir:
    :return:
    """
    
    dict_mode={1:'_poly',0:'_bbox'}
    folder = os.listdir(root_dir)
    dirs = [os.path.join(root_dir, f) for f in folder if os.path.isdir(os.path.join(root_dir, f))]
    
    dirs_smallpath = [f for f in folder if os.path.isdir(os.path.join(root_dir, f))]
    if 'ann' in dirs_smallpath and 'img' in dirs_smallpath and (len(dirs_smallpath) > 4 ):
        raise Exception("Expected 'ann_bbox', 'ann_poly','ann' and 'img' folders only, but more than one was found."
                        "Folder: {}".format(root_dir))

    elif 'ann'in dirs_smallpath and 'img' in dirs_smallpath:
        dirs = sorted(dirs)
        list_dir+=[dirs[-1]]
        correct_files(dirs[-1], dirs[0])
        correct_classes(dirs[0],os.path.join(os.path.split(dirs[0])[0], 'ann'+dict_mode[polygon]), polygon,class_list=class_list)
    else:
        for dir in dirs:
            iterate_folder(dir,polygon,list_dir,class_list)
        return list_dir


def correct_files(path_img: str, path_ann: str) -> None:
    """
    Correct files: create a Loop that fix all files
    """
    files = list(sorted(os.listdir(path_ann)))
    files_set = set()
    print(path_img)
    for f in tqdm(files):
        check_f = os.path.join(path_img, f)
        if  not path.exists(check_f[:-5]):
            print("File json not found in IMG - %s" % check_f)
            if check_f[-5:] == "json0":
                print("File fixed (json0) - %s" % check_f)
                # remove accents and spaces - ann
                f_improved_ann = remove_accent_space(f)
                # remove accents and spaces - img
                f_improved_img = remove_accent_space(f[:-6])
                if f_improved_ann not in files_set:
                    os.rename(os.path.join(path_ann, f), os.path.join(path_ann, f_improved_ann[:-1]))
                    os.rename(os.path.join(path_img, f[:-6]), os.path.join(path_img, f_improved_img))
                else:
                    print("Files with the same name : %s" % f_improved_ann)
                    print("Manually correction required")
                files_set.add(f_improved_ann)
            if '-' in f:
                print("File fixed  - %s" % check_f)
                # remove accents and spaces - ann
                f_improved_ann = remove_accent_space(f)
                # remove accents and spaces - img
                f_improved_img = remove_accent_space(f[:-5])
                if f_improved_ann not in files_set:
                    os.rename(os.path.join(path_ann, f), os.path.join(path_ann, f_improved_ann))
                    #os.rename(os.path.join(path_img, f[:-5]), os.path.join(path_img, f_improved_img))
                else:
                    print("Files with the same name : %s" % f_improved_ann)
                    print("Manually correction required")
                files_set.add(f_improved_ann)
        else:
            # remove accents and spaces - ann
            f_improved_ann = remove_accent_space(f)
            # remove accents and spaces - img
            f_improved_img = remove_accent_space(f[:-5])
            if f_improved_ann not in files_set:
                try:
                  os.rename(os.path.join(path_img, f[:-5]), os.path.join(path_img, f_improved_img))
                  os.rename(os.path.join(path_ann, f), os.path.join(path_ann, f_improved_ann))
                except:
                    print("Problem with the file name : %s" % os.path.join(path_ann, f))
                    print("Manually correction required")
                    print("Probably best solution will be to delete this file")
            else:
                print("Files with the same name : %s" % f_improved_ann)
                print("Manually correction required")
            files_set.add(f_improved_ann)


def remove_accent_space(s: str) -> str:
    """
    :param s: String - name of the file
    :return: String without accent, space and special character
    """
    s = s.strip()
    s = s.replace(" ", "_")
    s = s.replace("'", "_")
    s = s.replace("=", "_")
    s = s.replace("-", "_")
    s = unidecode.unidecode(s)
    return s


def correct_cocodataset_json(path_coco_dataset: str) -> None:
    """
    Correct cocodataset json: corrects a coco_dataset created before the dataset was clean
    :param path_json: path to the coco_dataset
    :return:
    """
    with open(path_coco_dataset) as json_file:
        data = json.load(json_file)
        json_file.close()
        imgs = data["images"]
        for img in tqdm(imgs):
            img["file_name"] = remove_accent_space(img["file_name"])
        with open(path_coco_dataset, 'w') as fp:
            json.dump(data, fp)
            fp.close()


def separate_train_test_ann(path_ann: str, p: float = 0.75) -> None:
    """
    :param path_ann: path to ann
    :param p: percentage related to the training size
    :return:This function will SPLIT this ann folder into two folders,
            one that contains train dataset (/ann/ann_train) and another that contains a test dataset (/ann/ann_test)
            all files will be moved to one of this folders
    """
    files = list(os.listdir(path_ann))
    p= (len([files[i] for i  in range(1,len(files)) if len(files[i])!=len(files[i-1])])+1)/len(files)
    test=[]
    train=[]
    for f in tqdm(files):
        f_path = os.path.join(path_ann, f)
        with open(f_path) as json_file:
            data = json.load(json_file)
            json_file.close()
            print(data)
            tags=data['tags'][0]['name']
            if (tags=='val'):
                test.append(f)
            else:
                train.append(f)
    os.makedirs(os.path.join(path_ann, "train"), exist_ok=True)
    os.makedirs(os.path.join(path_ann, "test"), exist_ok=True)
    
    for file in tqdm(train, maxinterval=len(train)):
        shutil.move(os.path.join(path_ann, file), os.path.join(path_ann, "train/"))
    for file in tqdm(test, maxinterval=len(test)):
        shutil.move(os.path.join(path_ann, file), os.path.join(path_ann, "test/"))
def separate_train_test_coco(path_coco: str, path_save_train: str,
                             path_save_test: str, p: float = 0.9) -> None:

    with open(path_coco) as json_file:
        coco_dict = json.load(json_file)
        json_file.close()

    train_coco = {'categories': coco_dict['categories'],
                  'licenses': coco_dict['licenses'],
                  'info': coco_dict['info'],
                  'images': [],
                  'annotations': []}

    test_coco = {'categories': coco_dict['categories'],
                 'info': coco_dict['info'],
                 'licenses': coco_dict['licenses'],
                 'images': [],
                 'annotations': []}

    images = coco_dict['images']
    annotations = coco_dict['annotations']
    p= (len([images[i]['file_name'] for i  in range(1,len(images)) if len(images[i]['file_name'])> 28 or len(images[i]['file_name'])!=len(images[i-1]['file_name'])])+1)/len(images)
    print([images[i] for i  in range(1,len(images)) if len(images[i]['file_name'])!=len(images[i-1]['file_name'])])
    train, test = train_test_split(images, train_size=p ,shuffle=False)

    for img in tqdm(train):
        train_coco['images'].append(img)
        objs = [f for f in annotations if f['image_id'] == img['id']]
        train_coco['annotations'].extend(objs)

    for img in tqdm(test):
        test_coco['images'].append(img)
        objs = [f for f in annotations if f['image_id'] == img['id']]
        test_coco['annotations'].extend(objs)

    with open(path_save_train, 'w') as fp:
        json.dump(train_coco, fp)
        fp.close()
    with open(path_save_test, 'w') as fp:
        json.dump(test_coco, fp)
        fp.close()


def clean_path(path_coco: str) -> None:
    with open(path_coco) as json_file:
        coco_dict = json.load(json_file)
        json_file.close()
    images = coco_dict['images']
    for img in tqdm(images):
        file = img["file_name"]
        file = file.replace(os.sep, posixpath.sep)
        img["file_name"] = file
    with open(path_coco, 'w') as fp:
        json.dump(coco_dict, fp)
        fp.close()


def correct_classes(path_ann: str,path_ann_save: str,polygon=True,class_list=[]) -> None:
    files = list(sorted(os.listdir(path_ann)))
    
    for f in files:
        f_path = os.path.join(path_ann, f)
        dir_save=path_ann_save
        f_path_ann_save= os.path.join(path_ann_save, f)
        with open(f_path) as json_file:
            data = json.load(json_file)
            json_file.close()
            objs = data["objects"]
            new_objs = []
            for obj in objs:
                if obj["classTitle"] == "Concrete bucket":
                    obj["classTitle"] = "Concrete_bucket"
                elif obj["classTitle"] == "Mobile crane":
                    obj["classTitle"] = "Mobile_crane"
                elif obj["classTitle"] == "Mixer truck":
                    obj["classTitle"] = "Mixer_truck"
                elif obj["classTitle"] == "Horizontal formwork":
                    obj["classTitle"] = "Horizontal_formwork"
                elif obj["classTitle"] == "Vertical formwork":
                    obj["classTitle"] = "Vertical_formwork"
                elif obj["classTitle"] == "Concrete pump hose":
                    obj["classTitle"] = "Concrete_pump_hose"
                elif obj["classTitle"] == "Orange_vest_people": #TNG : classe orange_vest_people n'est plus pertinente.
                    obj["classTitle"] = "People"
 
                if filterClasse(obj['geometryType'], obj["classTitle"],polygon,class_list):
                    new_objs.append(obj)
            data["objects"] = new_objs
            os.makedirs(dir_save, exist_ok=True)
            with open(f_path_ann_save, 'w') as fp:
                json.dump(data, fp)
                fp.close()


def filterClasse(geometry: str, title: str,polygon:bool=False,class_list=[]) -> bool:
    if (not polygon):
        
        if geometry != "polygon":
            if title in class_list : #and title != "Concrete pump hose" and title != "Lifting" and title != "Tower crane":
                return True
    else:
        if geometry != "rectangle" and (title == 'Shoring'or title =='Precast_wall' or title=='Concrete_pump_hose' or title == 'Vertical_formwork' or title== 'Rebars' or title=='Horizontal_formwork'):
            return True
    return False


if __name__ == "__main__":
    """
    INPUT
    - path_img: String - path to image folder
    - path_ann: String - path to json folder
    """
    path_img = "/home/leonardVM/train/L14/batch4/L14__sensor4/img"
    path_ann = "/home/leonardVM/train/L14/batch4/L14__sensor4/ann"
    #clean_path("../../../Desktop/main_model_bbox/coco_dataset.json")
    separate_train_test_ann("/data/chronsite-training/datasets_train/sensors_spindle/SPINDLE_STEREO_001-Noisy_Abraxas_augmented/ann/")
    #separate_train_test_coco("/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/coco_dataset.json",
    #                       "/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/coco_train.json",
    #                        "/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/coco_test.json")
            
    #iterate_folder("/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/")
    #correct_files(path_img, path_ann)
