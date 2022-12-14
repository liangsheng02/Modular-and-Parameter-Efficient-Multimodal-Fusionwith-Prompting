# [Modular-and-Parameter-Efficient-Multimodal-Fusionwith-Prompting](https://arxiv.org/abs/2203.08055)

# Dataset Preparation
Used datsets: Visual Question Answering v2 (VQAv2), and Multimodal Sarcasm Detection (MUStARD).

We do not distribute datasets because of the license issue.
Please download the datasets by yourself.
We use `pyarrow` to serialize the datasets, conversion scripts are located in `./utils/write_*.py`.
Please organize the datasets as follows and run `make_arrow` to convert the dataset to pyarrow.

## VQAv2
https://visualqa.org/download.html
Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip))

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...  
    ├── test2015              
    |   ├── COCO_test2015_000000000001.jpg
    |   └── ...         
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_OpenEnded_mscoco_test2015_questions.json
    ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json

```python
from .utils.write_vqa import make_arrow
make_arrow(root, arrows_root)
```

## MUStARD
https://github.com/soujanyaporia/MUStARD
Download Video clips and annotation file sarcasm_data.json and do as follows in command line.

### Extract Audio from Video
**Requisites**: [FFmpeg](https://ffmpeg.org/).
```ruby
# test: for %f in (*.mp4) do echo %f
for %f in (*.mp4) do ffmpeg -i "%f" -f wav -vn "%~nf.wav"
# command line extract .wav from videos
# convert stereo to mono: -ac; Resample to 16000: -ar
# for %f in (*.mp4) do ffmpeg -i "%f" -f wav -vn -ac 1 -ar 16000 "%~nf.wav"
```

### Vocal Separation
```ruby
# for %f in (*.wav) do python vocal_separation.py --file %f 
```

### Video Frames Extraction
**Requisites**: [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).
```ruby
# Extract from all videos
# Test: for %f in (./Dataset/MUStARD/*.mp4) do echo %f
for %f in (./Dataset/video/*.mp4) do FeatureExtraction.exe -f "./Dataset/video/%f"
```

    root
    ├── mustard_v            
    │   ├── 1_60
    │   │	├── frame_0.jpg	
	|	|	└── ...
    |   └── ...
    ├── mustard_a              
    |   ├── 1_60.wav
    |   └── ...  
    └── sarcasm_data.json

```python
from .utils.write_mmsd import make_arrow
make_arrow(root, arrows_root)
```# Modular-and-Parameter-Efficient-Multimodal-Fusionwith-Prompting
# Modular-and-Parameter-Efficient-Multimodal-Fusionwith-Prompting
