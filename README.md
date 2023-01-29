# Image_Colorization_with_Captioning
Repository for a "Pattern Recognition" assignment that uses Pytorch.

## INSTRUCTIONS
- Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/`.
- Download [training images](http://images.cocodataset.org/zips/train2014.zip), [validation images](https://drive.google.com/file/d/1AE8M8CsNpH_VaiHXI5B0Jef8FaW619YO/view?usp=share_link) and [test images](https://drive.google.com/file/d/1UBZmizwIS_wNYljEdnsLs3Q5f4kwc62p/view?usp=share_link) and unzip to `data/`.

### Train Image Captioning
- Run `parse_coco.py`
- Run `captioning_train.py`

### Generate Captions
- To generate captions without training the model, download the [Captioning checkpoint](https://drive.google.com/file/d/1tP76-ii6Llcaqd7DqTtBa0Fpk2rETIX3/view?usp=share_link) to `checkpoints_captioning/`.
- Run `caption_generator.py`

### Train Colorizer
- Run `colorizer_train.py`

### Generate Colored Images
- To generate colored images without training the model, download the [Colorizer checkpoint](https://drive.google.com/file/d/1jzI9pW4Izdu_0U1LBp12I25P_6pI_OUZ/view?usp=share_link) to `checkpoints_colorizer/`.
- Run `colorizer_inference.py`

### Train Combiend Model
- Download the [Captioning checkpoint](https://drive.google.com/file/d/1tP76-ii6Llcaqd7DqTtBa0Fpk2rETIX3/view?usp=share_link) to `checkpoints_captioning/`.
- Download the [Colorizer checkpoint](https://drive.google.com/file/d/1jzI9pW4Izdu_0U1LBp12I25P_6pI_OUZ/view?usp=share_link) to `checkpoints_colorizer/`.
- Run `combined_model_train.py`

### Generate Colored Images with the Combined Model
- Download the [Captioning checkpoint](https://drive.google.com/file/d/1tP76-ii6Llcaqd7DqTtBa0Fpk2rETIX3/view?usp=share_link) to `checkpoints_captioning/`.
- Download the [Colorizer checkpoint](https://drive.google.com/file/d/1jzI9pW4Izdu_0U1LBp12I25P_6pI_OUZ/view?usp=share_link) to `checkpoints_colorizer/`.
- To generate colored images without training the model, download the [Colorizer Combined checkpoint](https://drive.google.com/file/d/1SYza54vm13bc2fZkwlyWYn6MBGlHLA9_/view?usp=share_link) to `checkpoints_combined_model/`.
- Run `combined_model_inference.py`

### Calculate MSE, SSIM and PSNR Metrics for both Colorizer Models
- Download the [Captioning checkpoint](https://drive.google.com/file/d/1tP76-ii6Llcaqd7DqTtBa0Fpk2rETIX3/view?usp=share_link) to `checkpoints_captioning/`.
- Download the [Colorizer checkpoint](https://drive.google.com/file/d/1jzI9pW4Izdu_0U1LBp12I25P_6pI_OUZ/view?usp=share_link) to `checkpoints_colorizer/`.
- Download the [Colorizer Combined checkpoint](https://drive.google.com/file/d/1SYza54vm13bc2fZkwlyWYn6MBGlHLA9_/view?usp=share_link) to `checkpoints_combined_model/`.
- Run `evaluation.py`


## Citations
- Captioning model taken from: [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption).
