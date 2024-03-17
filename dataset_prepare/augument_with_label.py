import cv2
import os
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm

def load_yolo_boxes(filename, shape):
    with open(filename) as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        x1 = (x_center - width / 2) * shape[1]
        y1 = (y_center - height / 2) * shape[0]
        x2 = (x_center + width / 2) * shape[1]
        y2 = (y_center + height / 2) * shape[0]

        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_id))

    return BoundingBoxesOnImage(boxes, shape=shape)

def save_yolo_boxes(bbs, filename, shape):
    with open(filename, 'w') as f:
        for bb in bbs.bounding_boxes:
            x_center = (bb.x1 + bb.x2) / 2 / shape[1]
            y_center = (bb.y1 + bb.y2) / 2 / shape[0]
            width = (bb.x2 - bb.x1) / shape[1]
            height = (bb.y2 - bb.y1) / shape[0]

            f.write(f'{int(bb.label)} {x_center} {y_center} {width} {height}\n')

# source_dir = 'path/to/source'
# output_dir = 'path/to/output'
source_dir = r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\datasets\testdata'

output_dir = r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\datasets\testdata\1'

images = glob.glob(os.path.join(source_dir, '*.jpg'))

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # 尽管小概率，但是务必组合一些其他的扩充方式，这样可以确保标注的持久性
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.7, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
    iaa.Sometimes(0.9,iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
])

for image_path in tqdm(images):
    image = cv2.imread(image_path)
    bbs = load_yolo_boxes(image_path.replace('.jpg', '.txt'), image.shape)

    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    cv2.imwrite(os.path.join(output_dir, 'augument_random' + os.path.basename(image_path)), image_aug)
    save_yolo_boxes(bbs_aug, os.path.join(output_dir, 'augument_random' + os.path.basename(image_path).replace('.jpg', '.txt')), image_aug.shape)

print('Done')