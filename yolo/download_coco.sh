#!/bin/sh
wget -P coco http://images.cocodataset.org/zips/val2017.zip
unzip coco/val2017.zip -d coco
rm -f coco/val2017.zip
