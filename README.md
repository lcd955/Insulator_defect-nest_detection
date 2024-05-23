# Insulator_defect-nest_detection
基于YOLOv5网络的输配电线路故障检测模型研究，本项目基于yolov5构建而成
## 1、对于绝缘子缺陷检测

各个网络模型已经存放于models文件夹

## 2、标注格式转换方法

请看label_format_coversion文件夹。

## 3、对网上公开数据集的整合，预处理方法，包括图片加雾，添加噪声，随机裁剪方法        

  数据集预处理方法可以看dataset_prepare文件夹

## 4、开源数据集如下所示：
飞桨网址：https://aistudio.baidu.com/datasetdetail/270697/0

## 5、可视化特征图方法

参考yolov5_gradcam.py函数

