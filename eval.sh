#!/bin/bash
#! coding=utf-8
((epoch = 15000))

while ((epoch <= 30000))
do
  echo ${epoch}
  CUDA_VISIBLE_DEVICES=3 python -u tools/eval.py -c configs/yolov3_darknet.yml -o weights=output/yolov3_darknet/${epoch} --output_eval evaluation/
  python tools/my_utils.py
  ((epoch += 5000))
done