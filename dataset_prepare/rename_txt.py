import os
import glob

# 在当前目录下获取所有txt文件
# filepath=[r"D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\datasets\foggy_labels/"]

folder1 = r"D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\datasets\foggy_labels/"  # 需要修改名称的txt文件的目录
folder2 = r'D:\desk\yolov5\insulator_defect\InsulatorDataSet-master\datasets\foggy_img/'  # 参考的jpg文件的目录

# 获取所有.txt和.jpg文件
txt_files = sorted(glob.glob(os.path.join(folder1, "*.txt")))
jpg_files = sorted(glob.glob(os.path.join(folder2, "*.jpg")))

# 检查两个文件夹中的文件数量是否相同
if len(txt_files) != len(jpg_files):
    print("文件数量不匹配！")
else:
    # 遍历每个txt文件，根据jpg文件进行重命名
    for txt_file, jpg_file in zip(txt_files, jpg_files):
        # 提取jpg文件的基础文件名（没有扩展名）
        base_name = os.path.splitext(os.path.basename(jpg_file))[0]
        # 创建新的txt文件名
        new_name = "{}.txt".format(base_name)
        new_name_path = os.path.join(folder1, new_name)
        # 重命名txt文件
        os.rename(txt_file, new_name_path)