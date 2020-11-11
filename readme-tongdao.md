

原始数据标注文件anns.zip，原始图片文件imgs01.zip、imgs02.zip、imgs03.zip、imgs04.zip

将训练数据放在../data/paddle_tongdao下

```
data/paddle_tongdao
|--anns
|--imgs
```
创建临时变量，执行init.sh，将训练数据准备好
```
export UNZIP_PSW=xxx
chmod +x init.sh
./init.sh
``` 

使用tools/my_utils.py进行数据处理
1. 清洗原始数据集，只保留机械目标样本。def filter_invalid_data()
2. 将xml格式转换成coco格式。def xmltococo()
2. 统计基本信息，包括类别分类，各类别图片、样本数量。def statistic_class_number(xml_path='../data/paddle_tongdao/anns', flag=None)

训练
```
python -u tools/train.py -c configs/yolov3_darknet.yml
```
测试
```
CUDA_VISIBLE_DEVICES=0 python -u tools/eval.py -c configs/yolov3_darknet.yml \
-o weights=output/yolov3_darknet/model_final
```
预测
```
python -u tools/infer.py -c configs/yolov3_darknet.yml \
--infer_dir=../data/paddle_tongdao/val \
--output_dir=infer_output/ \
--draw_threshold=0.1 \
-o weights=output/yolov3_darknet/2600
```