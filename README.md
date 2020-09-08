# object_detection_tflite

Object Detection using TensorFlow Lite models.

FEATURES:
- Object detection for streaming video shot by (MacBook, RaspberryPi) Camera Module
- Fast object detection using Google Coral Edge TPU
- You can use YOLO V3 and YOLO V4
- Object detection for pre-recorded videos and photos
- Face detection and age,gender estimation

## setup of object detection

- (Optional: RaspberryPi) prepare RaspberryPi and RaspberryPi Camera Module
- install Python 3.7
- install TensorFlow 2.1.0
    - cf. https://www.tensorflow.org/install/pip
- install TensorFlow lite runtime
    - cf. https://www.tensorflow.org/lite/guide/python
    - you can know your platform of RaspberryPi with `> uname -a`
- (Optional: TPU) prepare Google Coral Edge TPU USB Accelerator
- (Optional: TPU) install Edge TPU runtime
    - cf. https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime
- (Optional: RaspberryPi & TPU) setup Edge TPU
    - create a new file `/etc/udev/rules.d/99-edgetpu-accelerator.rules` with the following contents
    ```
    SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",GROUP="plugdev"
    SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",GROUP="plugdev"
    ```
    - `> sudo usermod -aG plugdev pi` (if you use the default account of RaspberryPi "pi")
    - reboot RaspberryPi
    - check Edge TPU is recognized by RaspberryPi (`> lsusb`)
        - `ID 18d1:9302 Google Inc.`
- (Optional: RaspberryPi) install picamera
    - `> pip3 install picamera`
- install the rest of required Python packages
    - `> pip3 install -r requirements.txt`
- download pretrained TFlite weights
    - `> ./download_models.sh`
- If you want to use YOLO, see [Readme of YOLO](https://github.com/tetutaro/object_detection_tflite/blob/master/yolo/README.md)
    - you can use YOLO V3 tiny, YOLO V3 and YOLO V4

## object detection

- `> detect.py [OPTIONS]`
- OPTIONS:
    - `--tpu/--no-tpu`: use (or don't use) TPU (default: `--no-tpu`)
    - `--model <model>`: model to use (default: `ssd`)
        - `ssd`: `models/mobilenet_ssd_v2_coco_quant_postpresss*.tflite`
        - `face`: `models/mobilenet_ssd_v2_face_quant_postpresss*.tflite`
        - `yolov3-tiny`: `yolo/yolov3-tiny*.tflite`
        - `yolov3`: `yolo/yolov3*.tflite`
        - `yolov4`: `yolo/yolov4*.tflite`
    - `--target <target>`: what to detect (default: `all`)
        - `all`: all objects written in `models/coco_labels.txt`
        - you can indicate one object which is written in `models/coco_labels.txt` (cf. `bird`, `person`, ...)
    - `--threshold <threshold>`: threshold of probability which shown in otput (default: 0.5)
    - `--width <width>`: width of captured image (default: 1280)
    - `--height <height>`: height of captured image (default: 720)
    - `--hflip/--no-hflip`: flip image horizontally (default: True (RaspberryPi) False (MacOS))
    - `--vflip/--no-vflip`: flip image vertically (default: True (RaspberryPi) False (MacOS))
    - `--fontsize <fontsize>`: fontsize of text written within the output image (default: 20)
    - `--media <filename>`: pre-recorded video or photo (default: None)
    - `--fastforward <skip>`: skip some frames (pre-recorded video only) (default: 1(no skip))

## face detection and age, gender estimation

- `> face_agender.py [OPTIONS]`
- OPTIONS are the same as above (`model` and `target` are invalidated)

## setup of agender model

- see [Readme of Agender](https://github.com/tetutaro/object_detection_tflite/blob/master/agender/README.md)

## motion detection and object detection

- `> motion_detect.py [OPTIONS]`
- OPTIONS are the same as above (`model` and `target` are invalidated)
- models are...
    - `mobilenet`: `models/mobilenet_v2_1.0_224_quant*.tflite`
    - `bird`: `models/mobilenet_v2_1.0_224_inat_bird_quant*.tflite`
    - `insect`: `models/mobilenet_v2_1.0_224_inat_insect_quant*.tflite`
    - `plant`: `models/mobilenet_v2_1.0_224_inat_plant_quant*.tflite`
