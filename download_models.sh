#!/bin/bash
if [ ! -d models ]; then
    mkdir -p models
    wget https://dl.google.com/coral/canned_models/all_models.tar.gz
    tar -C models -xvzf all_models.tar.gz
    rm -f all_models.tar.gz
    wget https://launchpad.net/takao-fonts/trunk/15.03/+download/TakaoFonts_00303.01.zip
fi
if [ -f TakaoGothic.ttf ]; then
    exit
fi
unzip TakaoFonts_00303.01.zip
mv TakaoFonts_00303.01/TakaoGothic.ttf .
rm -rf TakaoFonts_00303.01
rm -f TakaoFonts_00303.01.zip
