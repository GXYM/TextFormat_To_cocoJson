This is a python3 example showing how to build a totaltext dataset with .txt format for abcnet training. 

Step one: Given polygonal annotation, generating bezier curve annotation.
    python Bezier_generator2.py

Step two: Given bezier curve annotation, generating coco-like annotation format for training abcnet. (Same as CTW1500)
    python generate_abcnet_json.py ./ train 0