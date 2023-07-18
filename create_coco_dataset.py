"""
VINCI Construction

Author: Iago Martinelli Lopes (iago.martinelli-lopes@vinci-construction.com)
Tutor: Djamil Yahia-Ouahmed
Date: 11/06/2020
Model : COCO dataset
Objective : In order to use Detectron2, it is necessary to create a JSON file
            which will be used by it. This JSON file join all json files provided by
            Supervisely, and you can find its specifications in the following link
            https://detectron2.readthedocs.io/tutorials/datasets.html.
"""

# MAIN function with INPUTS and INSTRUCTIONS is located in the end of this file

import os
import json
import posixpath

from typing import Dict, List

from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True



def iterate_folder_create_coco(main_path: str, root_dir: str, coco_dict: Dict, classes_list: List[str],
                               polygon:bool=True, desc = ""):
    """
    :param main_path:
    :return:
    """
    dict_mode={1:'_poly',0:'_bbox'}
    folder = os.listdir(root_dir)
    dirs = [os.path.join(root_dir, f) for f in folder if os.path.isdir(os.path.join(root_dir, f))]
    dirs_smallpath = [f for f in folder if os.path.isdir(os.path.join(root_dir, f)) and 'coco' not in f]

    if 'ann'+dict_mode[polygon] in dirs_smallpath and 'img' in dirs_smallpath and len(dirs_smallpath) > 4:
        raise Exception("Expected 'ann_bbox', 'ann_poly','ann' and 'img' folders only, but more than one was found."
                        "Folder: {}".format(root_dir))

    elif  'ann'+dict_mode[polygon]  in dirs_smallpath and 'img' in dirs_smallpath:
        dirs = sorted(dirs)
        if False :
            print(dirs)
        create_coco_dataset_iter(main_path, dirs[-1],
                                 os.path.join(os.path.split(dirs[0])[0],'ann'+dict_mode[polygon]),
                                 coco_dict, classes_list, desc = desc)
    else:
        for dir in dirs:
            iterate_folder_create_coco(main_path, dir, coco_dict, classes_list,polygon, desc= desc)


def create_coco_dataset_iter(main_path: str, img_dir: str, json_dir: str, coco_dict: Dict, classes_list: List[str],
                             desc = ""
                             ) -> None:
    # json
    json_files = list(sorted(os.listdir(json_dir)))
    obj_id = len(coco_dict['annotations']) + 1
    img_id = len(coco_dict['images']) + 1
    imgs = []  # only the name
    objs = []
    number_of_errors = 0
    
    for json_f in tqdm(json_files, desc= desc):
        img_name = json_f[:-5]
        box_path = os.path.join(json_dir, json_f)
        with open(box_path) as json_file:
            data = json.load(json_file)
            json_file.close()
            # images
            record_img = {}
            filename = img_name
            # check
            
            try:
                img = Image.open(os.path.join(img_dir, filename))
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(e)
                continue;
            width, height = img.size
            local_img_path = os.path.relpath(os.path.join(img_dir, filename), main_path)
            local_img_path = local_img_path.replace(os.sep, posixpath.sep)
            record_img["file_name"] = local_img_path
            record_img["id"] = img_id
            record_img["height"] = height
            record_img["width"] = width
            imgs.append(record_img)
            annos = data['objects']
            # annotations
            for b in annos:
                obj = {}
                obj['id'] = obj_id
                if b["geometryType"] == "rectangle":
                    points = b['points']['exterior']
                    try :
                        xmin, ymin = points[0]
                        xmax, ymax = points[1]
                        obj['segmentation'] = []
                        obj['area'] = (xmax - xmin) * (ymax - ymin)
                    except : #Exception dans le cas où les points se situent en dehors de l'image suite à la data
                        # augmentation
                        number_of_errors += 1
                        continue
                elif b["geometryType"] == "polygon":
                    points = b['points']['exterior']
                    try :
                        xmax = max(p[0] for p in points)
                        xmin = min(p[0] for p in points)
                        ymax = max(p[1] for p in points)
                        ymin = min(p[1] for p in points)
                        seg = [coord for p in points for coord in p]
                        obj['segmentation'] = [seg]
                        obj['area'] = (xmax - xmin) * (ymax - ymin)
                    except : #Exception dans le cas où les points se situent en dehors de l'image suite à la data
                        # augmentation
                        number_of_errors += 1
                        continue
                else:
                    raise Exception("Geometry type not yet implemented")
                obj['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]  # [x, y, w, h]
                obj['image_id'] = img_id
                obj['category_id'] = classes_list.index(b['classTitle'])
                obj['iscrowd'] = 0
                objs.append(obj)
                obj_id += 1
        img_id += 1
    coco_dict['images'].extend(imgs)
    coco_dict['annotations'].extend(objs)
    print("number_of_errors : ", number_of_errors)


def add_meta(classes_list: list, coco_dict: Dict,polygon:bool=False) -> List[str]:
    # meta data
    classes_list.insert(0, "background")
    # categories
    categories = []
    for i in range(1, len(classes_list)):
        cat = {}
        cat['Supercategory'] = 'none'
        cat['name'] = classes_list[i]
        cat['id'] = i
        categories.append(cat)
    coco_dict['categories'] = categories

    # info
    info = {}
    info['description'] = "dataset"
    info['year'] = 2023
    coco_dict['info'] = info

    # licenses
    licenses = []
    coco_dict['licenses'] = licenses

    coco_dict['images'] = []
    coco_dict['annotations'] = []
    return classes_list


def create_coco_dataset(meta_file: str, img_dir: str, json_dir: str, file_dir: str) -> Dict:
    """
    Create custom COCO dataset : join all .json file together into one in COCO dataset format

    :param meta_file: path to the meta file
    :param img_dir: path to the images
    :param json_dir: path to the jsons
    :param file_dir: path where COCO dataset will be saved
    :return: Dictionary which contains COCO dataset
    """
    # COCO dictionary
    coco_dict = {}

    # meta data
    with open(meta_file) as json_file:
        data = json.load(json_file)
        classes_list = data['classes']
        classes_list = [c['title'] for c in classes_list]
    classes_list.insert(0, 'background')

    # categories
    categories = []
    for i in range(1, len(classes_list)):
        cat = {}
        cat['Supercategory'] = 'none'
        cat['name'] = classes_list[i]
        cat['id'] = i
        categories.append(cat)
    coco_dict['categories'] = categories

    # json
    json_files = list(sorted(os.listdir(json_dir)))

    # info
    info = {}
    info['description'] = "Chronsite dataset"
    info['year'] = 2020
    coco_dict['info'] = info

    # licenses
    licenses = []
    coco_dict['licenses'] = licenses
    obj_id = 1

    imgs = []  # only the name
    objs = []
    for idx, json_f in tqdm(enumerate(json_files)):
        img_name = json_f[:-5]
        box_path = os.path.join(json_dir, json_f)
        with open(box_path) as json_file:
            data = json.load(json_file)
            json_file.close()
            # images
            record_img = {}
            filename = img_name
            # check
            img = Image.open(os.path.join(img_dir, filename))
            try:
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(e)
            width, height = img.size
            record_img["file_name"] = filename
            record_img["id"] = idx
            record_img["height"] = height
            record_img["width"] = width
            imgs.append(record_img)
            annos = data['objects']
            # annotations
            for b in annos:
                obj = {}
                obj['id'] = obj_id
                if b["geometryType"] == "rectangle":
                    points = b['points']['exterior']
                    xmin, ymin = points[0]
                    xmax, ymax = points[1]
                    obj['segmentation'] = []
                    obj['area'] = (xmax - xmin) * (ymax - ymin)
                elif b["geometryType"] == "polygon":
                    points = b['points']['exterior']
                    xmax = max(p[0] for p in points)
                    xmin = min(p[0] for p in points)
                    ymax = max(p[1] for p in points)
                    ymin = min(p[1] for p in points)
                    seg = [coord for p in points for coord in p]
                    obj['segmentation'] = [seg]
                    obj['area'] = (xmax - xmin) * (ymax - ymin)
                else:
                    raise ("Geometry type not yet implemented")
                obj['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]  # [x, y, w, h]
                obj['image_id'] = idx
                obj['category_id'] = classes_list.index(b['classTitle'])
                obj['iscrowd'] = 0
                objs.append(obj)
                obj_id += 1
    coco_dict['images'] = imgs
    coco_dict['annotations'] = objs
    try:
        with open(file_dir, 'w') as fp:
            json.dump(coco_dict, fp)
            fp.close()
        return coco_dict
    except:
        return coco_dict


def check_size(coco_path: str, img_path: str) -> None:
    """
    Function to check if all images sizes match with what is written at COCO dataset file
    :param coco_path:
    :param img_path:
    """
    with open(coco_path) as json_file:
        data = json.load(json_file)
        json_file.close()
        imgs = data["images"]
        for img_file in tqdm(imgs):
            i_p = os.path.join(img_path, img_file['file_name'])
            img = Image.open(i_p)
            try:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                if w != img_file['width'] or h != img_file['height']:
                    print("PROBLEM")
                    print(img_file['file_name'])
            except Exception as e:
                print(e)


if __name__ == "__main__":
    """
    INPUT
    - meta_file: String - path to meta file provided by supervisely
    - img_dir: String - path to image folder
    - json_dir: String - path to json folder
    - file_dir: String - path where coco_dataset file will be saved
    RECOMMENDATIONS
    1) CLEAN your dataset before creating COCO dataset
    2) VERIFY that meta from Train and Test dataset contains the 
       SAME values and in the same order
    3) YOU MUST create a COCO dataset to both Train and Test dataset
    """

    meta_file = "/data/leonardVM/automatisation/train/Batch3-batch4_poly/meta.json"
    # img_dir = "../../../Desktop/" \
    #           "chronsite_train_only_bbox/Batch1__Dataset_perspective/img"
    # json_dir = "../../../Desktop/" \
    #           "chronsite_train_only_bbox/Batch1__Dataset_perspective/ann"
    file_dir = "/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/coco_dataset.json" #to modify

    coco_dict = {}
    classes_list = add_meta(meta_file, coco_dict,True)
    iterate_folder_create_coco("/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/", "/data/leonardVM/automatisation/train/L14.2-cuchet__sensor4/", coco_dict, classes_list)
    with open(file_dir, 'w') as fp:
        json.dump(coco_dict, fp)
        fp.close()

    #  _ = create_coco_dataset(meta_file, img_dir, json_dir, file_dir)