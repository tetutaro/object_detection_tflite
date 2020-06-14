# object_detection_tflite

Object Detection using RaspberryPi Camera

## setup

- (Optional: RaspberryPi) prepare RaspberryPi and RaspberryPi Camera Module
- install Python 3.7
- install TensorFlow 2.1.0
    - cf. https://www.tensorflow.org/install/pip
- install tensorflow-addon
    - `> pip3 install tensorflow-addons==0.9.1`
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
- If you want to use YOLO, see [YOLO](/tetutaro/object_dtection_tflite/blob/master/yolo/README.md)

## detect object

- `> python3 detect.py [--tpu/--no-tpu] [--model <model>] [--target <target>] [--threshold <threshold>] [--width <width>] [--height <height>] [--hflip/--no-hflip] [--vflip/--no-vflip]`
    - `--tpu/--no-tpu`: use (or don't use) TPU (default: `--no-tpu`)
    - `--model <model>`: model to use (default: `coco`)
        - `coco`: `models/mobilenet_ssd_v2_coco_quant_postpresss*.tflite`
        - `face`: `models/mobilenet_ssd_v2_face_quant_postpresss*.tflite`
    - `--target <target>`: what to detect (default: `all`)
        - `all`: all objects written in `models/coco_labels.txt`
        - you can indicate one object which is written in `models/coco_labels.txt` (cf. `bird`, `person`, ...)
    - `--threshold <threshold>`: threshold of probability which shown in otput (default: 0.5)
    - `--width <width>`: width of captured image (default: 640)
    - `--height <height>`: height of captured image (default: 640)
    - `--hflip/--no-hflip`: flip image horizontally (default: True (RaspberryPi) False (MacOS))
    - `--vflip/--no-vflip`: flip image vertically (default: True (RaspberryPi) False (MacOS))
