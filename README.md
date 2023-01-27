# Image_Colorization_with_Captioning
Repository for a "Pattern Recognition" assignment that uses Pytorch.

## INSTRUCTIONS
- Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/`.
- Download [training images](http://images.cocodataset.org/zips/train2014.zip), [validation images](http://images.cocodataset.org/zips/val2014.zip) and [test images](http://images.cocodataset.org/zips/test2014.zip) and unzip to `data/`.

### Train
- Run `parse_coco.py`
- Run `train.py`

### Generate Captions
- To generate captions without training the model, downlaod the [Captioning checkpoint](https://drive.google.com/drive/folders/1uVuNwwoAZTdtsfwvYrqopBUY08KW-3tC?usp=sharing) named `coco_prefix_latest.pt` to `checkpoints_captioning/`.
- Run `caption_generator.py`

## Citations
- Captioning model taken from: [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption).
