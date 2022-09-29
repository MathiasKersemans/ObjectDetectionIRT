# Faster R-CNN for prediction of IRT signals

## 1. Install python environment：
Install [requirements.txt](https://github.com/MathiasKersemans/ObjectDetectionIRT/blob/main/requirements.txt) in a
[**Python>=3.10.0**](https://www.python.org/) environment.
```bash
git clone https://github.com/MathiasKersemans/ObjectDetectionIRT.git  # clone
pip install -r requirements.txt  # install
```

## 2. Clone the weights file
```bash
git lfs install
git clone https://huggingface.co/FelixTong/IRT_FasterRCNN
```
Put the weights file(`weights_file.pth`) to [weight](https://github.com/MathiasKersemans/ObjectDetectionIRT/tree/main/weights) directory.

## 3. Prepare your IRT processing images
The .jpg images are needed in this program.

Put your post-processing images(PCA, PPT, TSR...) in [example case](https://github.com/MathiasKersemans/ObjectDetectionIRT/tree/main/example%20case) directory.

## 4. Inference with Prediction.py
```bash
python Prediction.py --image_draw 'FBHs_PCA_1.jpg' # The image name used to draw the prediction results
```

## <div align="center">Contact</div>
Mathias Kersemans (Mathias.Kersemans@UGent.be)

Zongfei Tong (tongzongfei@outlook.com)
