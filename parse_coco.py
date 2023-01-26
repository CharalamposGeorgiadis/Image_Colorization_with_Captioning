import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
from IPython.display import clear_output


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_path = f"./data/oscar_split_ViT-B_32_train.pkl"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open('./data/train_caption.json', 'r') as f:
        data = json.load(f)

    clear_output()
    print("%0d captions loaded from json " % len(data))
    clear_output()
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./data/val2014/COCO_val2014_{int(img_id):012d}.jpg"

        image = io.imread(filename, as_gray=True) * 255
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device)
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(data_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(data_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    clear_output()
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))


if __name__ == '__main__':
    main()
