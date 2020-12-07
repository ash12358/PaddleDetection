#!/bin/bash
cd ~/data
rm -rf paddle_tongdao
mkdir paddle_tongdao
cd paddle_tongdao
mkdir anns
mkdir imgs
unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/anns/ ~/data/data59386/anns.zip
unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59386/imgs01.zip
unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59386/imgs02.zip
unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59386/imgs03.zip
unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59386/imgs04.zip

#xiaozhi
#unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/anns/ ~/data/data59684/anns.zip
#unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59684/imgs01.zip
#unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59684/imgs02.zip
#unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59684/imgs03.zip
#unzip -q -P $UNZIP_PSW -d ~/data/paddle_tongdao/imgs/ ~/data/data59684/imgs04.zip

# macos
# unzip -q -P 12358 -d /Users/ash/PycharmProjects/data/paddle_tongdao/anns/ /Volumes/Elements/PycharmProjects/data/anns.zip
# unzip -q -P 12358 -d /Users/ash/PycharmProjects/data/paddle_tongdao/imgs/ /Volumes/Elements/PycharmProjects/data/imgs01.zip
# unzip -q -P 12358 -d /Users/ash/PycharmProjects/data/paddle_tongdao/imgs/ /Volumes/Elements/PycharmProjects/data/imgs02.zip
# unzip -q -P 12358 -d /Users/ash/PycharmProjects/data/paddle_tongdao/imgs/ /Volumes/Elements/PycharmProjects/data/imgs03.zip
# unzip -q -P 12358 -d /Users/ash/PycharmProjects/data/paddle_tongdao/imgs/ /Volumes/Elements/PycharmProjects/data/imgs04.zip