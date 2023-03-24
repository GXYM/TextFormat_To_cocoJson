This is a python3 example showing how to build a custom dataset for abcnet training. The example image and annotation are from CTW1500 dataset (https://github.com/Yuliang-Liu/Curve-Text-Detector/tree/master/data)

Step one: Given polygonal annotation, generating bezier curve annotation.
    python Bezier_generator2.py

Step two: Given bezier curve annotation, generating coco-like annotation format for training abcnet.
    python generate_abcnet_json.py ./ train 0