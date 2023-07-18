"""
VINCI Construction

Author: Iago Martinelli Lopes (iago.martinelli-lopes@vinci-construction.com)
Tutor: Djamil Yahia-Ouahmed
Date: 15/06/2020
Objective : Evaluate DL models using Detectron2 backend

THIS FUNCTION ONLY WORK IN THE VIRTUAL MACHINE - CUDA AVAILABLE
"""

# MAIN function with INPUTS and INSTRUCTIONS is located in the end of this file

import os

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import logging


def evaluate(model: str, test_dataset: str,
             num_classes: int, save_dir: str, load_dir: str) -> None:
    """
    Explanations of the parameters are located in the MAIN function

    :param model: String
    :param train_dataset: String
    :param test_dataset: String
    :param num_classes: int
    :param save_dir: String
    :param load_dir: String
    """
    if False :
        print("MODEL")
        print()
    cfg = get_cfg()
    if model == "Faster RCNN":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    elif model == "Mask RCNN":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    elif model == "Retina Net":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (test_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = load_dir
    cfg.OUTPUT_DIR = save_dir

    trainer = DefaultTrainer(cfg,False)
    trainer.resume_or_load(resume=False)
    if False :
        print("Evaluation start")
    os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
    evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=os.path.join(save_dir, "eval"))
    val_loader = build_detection_test_loader(cfg, test_dataset)
    score_dict=inference_on_dataset(trainer.model, val_loader, evaluator)

    logging.info(f"Score_Dict : \n {score_dict}")
    return score_dict

def get_coco_dataset(dataset_dir: str, name: str, img_dir: str) -> None:
    """
    :param dataset_dir: Path to COCO dataset
    :param name: Name of the DATASET
    :param img_dir: path to the images
    """
    DatasetCatalog.clear()
    register_coco_instances(name, {}, dataset_dir, img_dir)
    test_metadata = MetadataCatalog.get(dataset_dir)


def evaluation_complet(
    model = "Faster RCNN" ,
    num_classes = 1,
    dir_coco_dataset_test = "/mnt/chronsite-training/datasets_train/La_valette_with_second_batch_5/coco_dataset_v077/coco_test_bbox.json",
    dir_img_test = "/mnt/chronsite-training/datasets_train/La_valette_with_second_batch_5/test/",
    save_dir = "/mnt/chronsite-training/models/La_valette_with_second_batch_5/bbox/",
    load_dir = "/mnt/chronsite-training/models/La_valette_with_second_batch_5/bbox/v077/model_0020999.pth"):

    print("Creating dataset")
    print()
    test_dataset = "test"
    DatasetCatalog.clear()
    get_coco_dataset(dir_coco_dataset_test, test_dataset, dir_img_test)
    print("Creating metadata")
    print()
    test_metadata = MetadataCatalog.get(test_dataset)
    os.makedirs(save_dir, exist_ok=True)
    score_dict=evaluate(model, test_dataset, num_classes, save_dir, load_dir)
    return score_dict