import argparse
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, help='The path to the input RGB image.')
    parser.add_argument('--output_folder', type=str, help='Where to save the segmentation masks to.')
    parser.add_argument('--scale_factor', type=int,
                        help='Input images are resized to (height / scale_factor, width / scale_factor)', default=1)

    args = parser.parse_args()

    assert args.scale_factor >= 1, f"Scale factor must be at least 1, got {args.scale_factor}."

    cfg = get_cfg()

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    dataset_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = dataset_metadata.thing_classes

    person_label = class_names.index('person')
    predictor = DefaultPredictor(cfg)

    os.makedirs(args.output_folder, exist_ok=True)

    image = Image.open(args.input_image)
    input_size = image.size

    image = image.resize((image.height // args.scale_factor, image.width // args.scale_factor))
    image = np.asarray(image)

    start = datetime.now()

    output = predictor(image)

    elapsed = datetime.now() - start
    print(f"Took {elapsed} to predict instance segmentation masks.")

    matching_masks = output['instances'].get('pred_classes') == person_label
    people_masks = output['instances'].get('pred_masks')[matching_masks]
    combined_masks = np.zeros_like(image, dtype=np.uint8)

    for mask in people_masks.cpu().numpy():
        combined_masks[mask] = 255

    mask = Image.fromarray(combined_masks).convert('L')
    mask = mask.resize(input_size)
    mask.save(os.path.join(args.output_folder, f"{Path(args.input_image).stem}_mask.png"))
