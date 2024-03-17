import cv2
import numpy as np
import math
import os
from tqdm import tqdm
#
# # 图像和标签框一起旋转
# def rotate_image_and_boxes(img, boxes, angle, scale=1.):
#     w, h = img.shape[1], img.shape[0]
#     cx, cy = w // 2, h // 2
#
#     M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
#     rotated_img = cv2.warpAffine(img, M, (w, h))
#
#     rotated_boxes = []
#     for box in boxes:
#         label,x, y, w, h = box
#         corners = np.array([
#             [x-w/2, y-h/2],
#             [x-w/2, y+h/2],
#             [x+w/2, y-h/2],
#             [x+w/2, y+h/2]
#         ])
#
#         corners = np.hstack((corners, np.ones((4, 1))))
#         corners = np.dot(M, corners.T).T
#         x_min, y_min = corners.min(axis=0)[:2]
#         x_max, y_max = corners.max(axis=0)[:2]
#
#         rotated_boxes.append([label,x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, x_max - x_min, y_max - y_min])
#
#     return rotated_img, rotated_boxes
#
# # 读取标记文件
# def read_annotation_file(file_path):
#     boxes = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             items = line.strip().split(" ")
#             class_id = int(items[0])
#             x, y, w, h = map(float, items[1:])
#             boxes.append([class_id, x, y, w, h])
#     return boxes
#
# img = cv2.imread(r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\images\001.jpg')
# boxes = read_annotation_file(r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\labels\worktxt\001.txt')
#
# rotated_img, rotated_boxes = rotate_image_and_boxes(img, boxes, angle=30)
#
# cv2.imwrite(r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\labels\rotated_image.jpg', rotated_img)
# with open(r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\labels\rotated_image.txt', 'w') as file:
#     for box in rotated_boxes:
#         file.write(" ".join(map(str, box)) + "\n")

def rotate_image_and_boxes(image, boxes):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 转换成点并应用旋转和转换反馈到bounding box的格式
    new_boxes = []
    for box in boxes:
        points = np.int0(cv2.transform(np.array([[
            [box[0], box[1]],
            [box[0] + box[2], box[1]],
            [box[0] + box[2], box[1] + box[3]],
            [box[0], box[1] + box[3]]
        ]]), M))
        new_box = cv2.boundingRect(points)
        new_boxes.append(new_box)

    return rotated, new_boxes

for image_file in tqdm(os.listdir(r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\images/')):
    image = cv2.imread(fr'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators/images/{image_file}')
    base_name = os.path.splitext(image_file)[0]
    box_file = fr'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\Defective_Insulators\worktxt/{base_name}.txt'
    with open(box_file, 'r') as f:
        boxes = []
        for line in f:
            elements = line.strip().split()
            x_center, y_center, box_w, box_h = map(float, elements[1:])
            x1 = (x_center - box_w / 2) * image.shape[1]
            y1 = (y_center - box_h / 2) * image.shape[0]
            x2 = x1 + box_w * image.shape[1]
            y2 = y1 + box_h * image.shape[0]
            boxes.append([x1, y1, x2-x1, y2-y1])
        rotated_image, new_boxes = rotate_image_and_boxes(image, boxes)
        # 保存旋转后的图像
        img_path=fr'D:\desk\yolov5\dataset_raw\rotated_images_30/{base_name}_rotated_30.jpg'
        base_img_path = os.path.dirname(img_path)
        if not os.path.exists(base_img_path):
            os.makedirs(base_img_path)  # 如果不存在，创建路径
        cv2.imwrite(img_path, rotated_image)
        # 保存旋转后的框，您可能会想要将这些框转换回YOLO格式
        out_file = fr'D:\desk\yolov5\dataset_raw\rotated_boxes_30/{base_name}_rotated_30.txt'

        base_out_path= os.path.dirname(out_file)

        if not os.path.exists(base_out_path):
            os.makedirs(base_out_path)  # 如果不存在，创建路径
        with open(out_file, 'w') as f:
            for box in new_boxes:
                x_center = (box[0] + box[2] / 2) / rotated_image.shape[1]
                y_center = (box[1] + box[3] / 2) / rotated_image.shape[0]
                box_w = box[2] / rotated_image.shape[1]
                box_h = box[3] / rotated_image.shape[0]
                # 写入类别标签，这里假设类别不变仍为0
                f.write(f'0 {x_center} {y_center} {box_w} {box_h}\n')