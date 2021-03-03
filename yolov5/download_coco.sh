#!/usr/bin/env bash
if [ -d "val2017" ]; then
    exit 0
fi
if [ ! -f "val2017.zip" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
fi
unzip val2017.zip
