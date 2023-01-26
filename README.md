# Image_Colorization_with_Captioning
Repository for a "Pattern Recognition" assignment that uses Pytorch.

## INSTRUCTIONS
- Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/`.
- Download [training images](http://images.cocodataset.org/zips/train2014.zip), [validation images](http://images.cocodataset.org/zips/val2014.zip) and [test images](http://images.cocodataset.org/zips/test2014.zip) and unzip to `data/`.
- Downlaod [checkpoints](https://drive.google.com/drive/folders/1uVuNwwoAZTdtsfwvYrqopBUY08KW-3tC?usp=sharing) and unzip to `checkpoints_captioning/`.

### Train
- Run `parse_coco.py`
- Run `train.py`

### Generate Captions
- Run `caption_generator.py`
- Change filename in **Line 41** in `caption_generator.py` to load different pretrained model weights. The `coco_prefix_xxx` and `coco_prefix_latest` models were trained using Grayscale images. The `coco_weights` model was trained using RGB images.

## Citations
- Captioning model taken from: [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption).
