# TextFormat_To_cocoJson

This is a python3 example showing how to build a detection txt format for COCO format train  

Step one: Given polygonal annotation, generating bezier curve annotation.  
    python generator2_xxxx.py

Step two: Given bezier curve annotation, generating coco-like annotation format for training abcnet. (Same as CTW1500)  
    python generate_json.py

