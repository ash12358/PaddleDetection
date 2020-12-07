#! coding=utf-8
import os, shutil
from xml.etree import ElementTree as ET
from tqdm import tqdm
import glob
import json
import numpy as np
import random
random.seed(0)


def statistic_class_number(xml_path='../data/paddle_tongdao/anns', flag=None):

    """
    统计各类别数量信息
    :param xml_path，xml文件存放路径
    :param flag, 根据flag判断要统计train还是val
    :Returns dict{
        class_name -> str: [number_of_obj -> int, number_of_img -> int]
    }
    """
    class_info = {}
    if flag is not None:
        flag = os.listdir(flag)
    for xml_name in os.listdir(xml_path):
        if not xml_name.endswith('.xml') or (flag is not None and xml_name[:-4]+'.jpg' not in flag):
            continue
        tree = ET.parse(os.path.join(xml_path, xml_name))
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_info:
                class_info[class_name] = [1, None]  # [目标数量, 图片数量]
                class_info[class_name][1] = set()
                class_info[class_name][1].add(xml_name)
            else:
                class_info[class_name][0] = class_info[class_name][0] + 1
                class_info[class_name][1].add(xml_name)

    # 按照目标数量从大到小排序
    class_info = sorted(class_info.items(), key=lambda x: -x[1][0])
    print('类别名称\t目标数量\t图片数量')
    for class_name, (number_of_obj, number_of_img) in class_info:
        print('{}\t{}\t{}'.format(class_name, number_of_obj, len(number_of_img)))


def select_mini_dataset():
    xml_path = '../data/paddle_tongdao/anns'
    img_path = '../data/paddle_tongdao/imgs'
    xml_names = [xml_name[:-4] for xml_name in os.listdir(xml_path) if xml_name.endswith('.xml')]
    img_names = [img_name[:-4] for img_name in os.listdir(img_path) if img_name.endswith('.jpg')]

    mini_batch = random.sample(xml_names, 5000)

    print('mini_batch len is ', len(mini_batch))

    xml_dels = []
    img_dels = []

    for xml_name in xml_names:
        if xml_name not in mini_batch:
            xml_dels.append(os.path.join(xml_path, xml_name+'.xml'))

    for img_name in img_names:
        if img_name not in mini_batch:
            img_dels.append(os.path.join(img_path, img_name+'.jpg'))
    xml_dels.sort()
    img_dels.sort()

    for path in xml_dels:
        os.remove(path)
    for path in img_dels:
        os.remove(path)
    print('删除xml {} 个'.format(len(xml_dels)))
    print('删除img {} 个'.format(len(img_dels)))


def filter_invalid_data():
    """
    修改xml，去除非机械目标
    Returns:

    """
    invalid_target = ['dustproof', 'colorbelts', 'smog', 'fire']
    xml_path = '../data/paddle_tongdao/anns'
    new_xml_path = xml_path + '0'

    if os.path.exists(new_xml_path):
        shutil.rmtree(new_xml_path)
    os.makedirs(new_xml_path)

    del_xml = []
    for xml_name in os.listdir(xml_path):
        if not xml_name.endswith('.xml'):
            continue
        shutil.copy(os.path.join(xml_path, xml_name), os.path.join(new_xml_path, xml_name))
        tree = ET.parse(os.path.join(new_xml_path, xml_name))
        root = tree.getroot()
        del_obj = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in invalid_target:
                del_obj.append(obj)
        for del_o in del_obj:
            root.remove(del_o)
        if len(root.findall('object')) == 0:
            del_xml.append(xml_name)
        tree.write(os.path.join(new_xml_path, xml_name), encoding='utf-8')

    for del_x in del_xml:
        os.remove(os.path.join(new_xml_path, del_x))


def xmltococo():
    start_bounding_box_id = 1

    def get(root, name):
        return root.findall(name)

    def get_and_check(root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
        if length == 1:
            vars = vars[0]
        return vars

    def convert(xml_list, json_file):
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
        categories = pre_define_categories.copy()
        bnd_id = start_bounding_box_id
        all_categories = {}
        for index, line in enumerate(xml_list):
            # print("Processing %s"%(line))
            xml_f = line
            tree = ET.parse(xml_f)
            root = tree.getroot()

            filename = os.path.basename(xml_f)[:-4] + ".jpg"
            image_id = 20190000001 + index
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
            json_dict['images'].append(image)
            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category in all_categories:
                    all_categories[category] += 1
                else:
                    all_categories[category] = 1
                if category not in categories:
                    if only_care_pre_define_categories:
                        continue
                    new_id = len(categories) + 1
                    print(
                        "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                            category, pre_define_categories, new_id))
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
                assert (xmax > xmin), "xmax <= xmin, {}".format(line)
                assert (ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()
        print("------------create {} done--------------".format(json_file))
        print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                      all_categories.keys(),
                                                                                      len(pre_define_categories),
                                                                                      pre_define_categories.keys()))
        print("category: id --> {}".format(categories))
        print(categories.keys())
        print(categories.values())


    classes = ['towercrane', 'digger', 'motocrane', 'pushdozer', 'truck', 'trailer', 'smalltruck', 'cementmixer', 'van', 'pumpcar', 'pilingmachine', ]
    path2 = '../data/paddle_tongdao'
    xml_dir = path2 + "/anns0"
    save_json_train = path2+'/annotations/train.json'
    save_json_val = path2+'/annotations/val.json'
    train_ratio = 0.9

    if os.path.exists(path2 + "/annotations"):
        shutil.rmtree(path2 + "/annotations")
    os.makedirs(path2 + "/annotations")

    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    xml_list = glob.glob(xml_dir + "/*.xml")
    xml_list = np.sort(xml_list)
    np.random.seed(100)
    np.random.shuffle(xml_list)

    train_num = int(len(xml_list) * train_ratio)
    xml_list_train = xml_list[:train_num]
    xml_list_val = xml_list[train_num:]

    convert(xml_list_train, save_json_train)
    convert(xml_list_val, save_json_val)

    if os.path.exists(path2 + "/train"):
        shutil.rmtree(path2 + "/train")
    os.makedirs(path2 + "/train")
    if os.path.exists(path2 + "/val"):
        shutil.rmtree(path2 + "/val")
    os.makedirs(path2 + "/val")

    for xml in xml_list_train:
        img_name = xml.split('/')[-1][:-4]+'.jpg'
        img = path2 + "/imgs/" + img_name
        shutil.copyfile(img, path2 + "/train/" + os.path.basename(img))

    for xml in xml_list_val:
        img_name = xml.split('/')[-1][:-4] + '.jpg'
        img = path2 + "/imgs/" + img_name
        shutil.copyfile(img, path2 + "/val/" + os.path.basename(img))

    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))


def area(xmin, ymin, xmax, ymax):
    return max(0, xmax-xmin) * max(0, ymax-ymin)

def iou(b1, b2):

    xmin_b1, ymin_b1, xmax_b1, ymax_b1 = b1
    xmax_b1 = xmax_b1 + xmin_b1
    ymax_b1 = ymax_b1 + ymin_b1

    xmin_b2, ymin_b2, xmax_b2, ymax_b2 = b2
    xmax_b2 = xmax_b2 + xmin_b2
    ymax_b2 = ymax_b2 + ymin_b2

    xmin = max(xmin_b1, xmin_b2)
    ymin = max(ymin_b1, ymin_b2)
    xmax = min(xmax_b1, xmax_b2)
    ymax = min(ymax_b1, ymax_b2)

    inter = area(xmin, ymin, xmax, ymax)
    union = area(xmin_b1, ymin_b1, xmax_b1, ymax_b1) + area(xmin_b2, ymin_b2, xmax_b2, ymax_b2) - inter

    return inter / union


def eval():
    category_to_id = {'towercrane': 1, 'digger': 2, 'motocrane': 3, 'pushdozer': 4, 'truck': 5, 'trailer': 6,
                      'smalltruck': 7, 'cementmixer': 8, 'van': 9, 'pumpcar': 10, 'pilingmachine': 11}
    id_to_category = {1: 'towercrane', 2: 'digger', 3: 'motocrane', 4: 'pushdozer', 5: 'truck', 6: 'trailer',
                      7: 'smalltruck', 8: 'cementmixer', 9: 'van', 10: 'pumpcar', 11: 'pilingmachine'}
    bbox_path = 'evaluation/bbox.json'
    val_path = '../data/paddle_tongdao/annotations/val.json'
    with open(bbox_path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        ground = json.load(f)
        ground = ground['annotations']

    # 过滤掉pred中score小于0.5的
    pred = list(filter(lambda x: x['score']>0.3, pred))

    TP_FP = {}
    TP = {}
    FN = {}

    # 计算TP和FN
    for obj_g in ground:
        category_id_g, bbox_g = obj_g['category_id'], obj_g['bbox']
        found = False
        # print(category_id_g, bbox_g)
        for obj_p in pred:
            category_id_p, bbox_p = obj_p['category_id'], obj_p['bbox']
            # print(category_id_p, bbox_p)
            if category_id_g == category_id_p and iou(bbox_g, bbox_p) > 0.5:
                TP[category_id_g] = TP.get(category_id_g, 0) + 1
                found = True
                break
        if found is False:
            FN[category_id_g] = FN.get(category_id_g, 0) + 1

    # 计算TP_FP
    for obj_g in ground:
        category_id_g, bbox_g = obj_g['category_id'], obj_g['bbox']
        TP_FP[category_id_g] = TP_FP.get(category_id_g, 0) + 1

    # print(TP)
    # print(TP_FP)
    # print(FN)

    accuracy = {}
    for category_id in id_to_category.keys():
        accuracy[id_to_category[category_id]] = TP.get(category_id, 0) / (TP_FP.get(category_id, 0) + FN.get(category_id, 0) + 0.00000001)
    print('accuracy')
    for category in accuracy.keys():
        print('{}\t{}'.format(category, accuracy[category]))
    print('='*30)


def tf_idf_weights(roidbs):
    """

    Args:
        roidbs:

    Returns:

    """
    TF = {}
    IDF = {}
    img_weights = []

    for i, roidb in enumerate(roidbs):
        img_cls = list([k for cls in roidbs[i]['gt_class'] for k in cls])
        help = set()
        for c in img_cls:
            TF[c] = TF.get(c, 0) + 1
            help.add(c)
        for c in help:
            IDF[c] = IDF.get(c, 0) + 1

    SUM_TF = 0
    SUM_IDF = 0
    for c in TF.keys():
        SUM_TF += TF[c]
    for c in IDF.keys():
        SUM_IDF += IDF[c]
    for c in TF.keys():
        TF[c] = TF[c] / SUM_TF
    for c in IDF.keys():
        IDF[c] = IDF[c] / SUM_IDF

    # print('\n\n TF IDF')
    # print(TF)
    # print(IDF)

    for i, roidb in enumerate(roidbs):
        weights = 0
        img_cls = set([k for cls in roidbs[i]['gt_class'] for k in cls])
        for c in img_cls:
            weights += ((1 / TF[c]) * (1 / IDF[c]))
        img_weights.append(weights)
    # probabilities sum to 1
    img_weights = img_weights / np.sum(img_weights)

    return img_weights


def sample_test():
    path = 'output/yolov3.reader'
    xml_path = '../data/paddle_tongdao/anns0'
    class_number = {}
    help = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.endswith('.jpg'):
                xml_name = line.split('/')[-1][:-4]+'.xml'
                if xml_name not in help:
                    help[xml_name] = {}
                    tree = ET.parse(os.path.join(xml_path, xml_name))
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        help[xml_name][class_name] = help[xml_name].get(class_name, 0) + 1

                for class_name in help[xml_name].keys():
                    class_number[class_name] = class_number.get(class_name, 0) + help[xml_name][class_name]

    print('采样的各类别目标数')
    class_number = sorted(class_number.items(), key=lambda x: -x[1])
    for class_name, cnt in class_number:
        print('{}\t{}'.format(class_name, cnt))


if __name__ == '__main__':
    # statistic_class_number('../data/paddle_tongdao/anns')

    # filter_invalid_data()
    # select_mini_dataset()
    # statistic_class_number('../data/paddle_tongdao/anns0')
    #
    # xml to coco
    # xmltococo()
    # statistic_class_number('../data/paddle_tongdao/anns0', '../data/paddle_tongdao/train')
    # statistic_class_number('../data/paddle_tongdao/anns0', '../data/paddle_tongdao/val')

    # eval()
    # print(iou([0,0,1,1], [0.5,0.5,1,1]))

    # sample_test()
    pass