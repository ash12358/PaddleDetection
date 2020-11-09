训练数据放在../data/paddle_tongdao下

使用tools/my_utils.py进行数据处理
1. 清洗原始数据集，只保留机械目标样本。def filter_invalid_data()
2. 将xml格式转换成coco格式。def xmltococo()
2. 统计基本信息，包括类别分类，各类别图片、样本数量。def statistic_class_number(xml_path='../data/paddle_tongdao/anns', flag=None)