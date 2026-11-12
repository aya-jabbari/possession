---
license: agpl-3.0
datasets:
- martinjolif/football-player-detection
base_model:
- Ultralytics/YOLO11m
pipeline_tag: object-detection
---

---
# version: 0.1.0
---

# Football Player Detection (YOLO11m)

This model is a fine-tuned **YOLO11m** object detector trained to identify key entities in football (soccer) match images.  
It was trained on the **[martinjolif/football-player-detection](https://huggingface.co/datasets/martinjolif/football-player-detection)** dataset and is suitable for sports analytics tasks such as player tracking or match analysis for example.

## Detected Classes

- player
- goalkeeper
- referee
- ball
  
## Training Details

- **Base model:** Ultralytics YOLO11m  
- **Task:** Object Detection  
- **Dataset:** martinjolif/football-player-detection  
- **License:** AGPL-3.0  

## Evaluation Results

Results on the test split using standard object detection metrics (Precision, Recall, F1, mAP50, mAP50-95):

| Class       | Images | Instances | Precision | Recall | F1-score | mAP50 | mAP50-95 |
|------------|--------|-----------|-----------|--------|----------|---------|--------------|
| ball       | 23     | 24        | 0.8912    | 0.6833 | 0.7735   | 0.6799  | 0.3380       |
| goalkeeper | 19     | 19        | 0.9473    | 0.9463 | 0.9468   | 0.9413  | 0.8024       |
| player     | 25     | 500       | 0.9576    | 0.9940 | 0.9755   | 0.9937  | 0.8737       |
| referee    | 25     | 56        | 0.9059    | 0.9643 | 0.9342   | 0.9888  | 0.7741       |

Prediction Examples:

<img src="figure-1.png" alt="figure1" width="800">
<img src="figure-2.png" alt="figure2" width="800">

## Notes

- Excellent performance on **player detection**, with very high recall and mAP.
- **Ball detection** is more challenging due to small object size and motion blur.

## Intended Use

- Football match analytics  
- Player, referee, and ball tracking  
- Annotation assistance for sports datasets  
- Research in sports computer vision  

## Limitations

- Performance may drop on non-broadcast footage, unusual camera angles, or low-resolution images.
- Trained specifically for football (soccer); not intended for other sports without further fine-tuning.

## Usage

```bash
wget https://huggingface.co/martinjolif/yolo-football-player-detection/blob/main/yolo-football-player-detection.pt
```

```bash
from ultralytics import YOLO

model = YOLO("yolo-football-player-detection.pt")
```

## Citation

If you use this model or the dataset, please cite the dataset authors and Ultralytics YOLO accordingly.
