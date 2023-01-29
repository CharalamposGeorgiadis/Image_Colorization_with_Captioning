import torch
import os
from captioning_model import ClipCaptionModel
import skimage.io as io
import PIL.Image
import clip
from torchvision import transforms
from combined_model import CombinedColorizationModel
from colorizer_model import Generator
from skimage.color import rgb2lab
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import skimage.io
from IPython.display import clear_output
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch.nn.functional as nnf
import sys

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Loading captioning model
prefix_length = 10
captioning_path = os.path.join('checkpoints_captioning/coco_prefix_latest.pt')
captioning_model = ClipCaptionModel(prefix_length).to(device)
captioning_model.load_state_dict(torch.load(captioning_path, map_location='cpu'))
captioning_model = captioning_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Loading colorization model
colorizer_path = os.path.join('checkpoints_colorizer/cp-107.pth')
cpkt = torch.load(os.path.join(colorizer_path), map_location=device)
colorization_model = Generator().to(device)
colorization_model.load_state_dict(cpkt['G_state_dict'])
colorization_model.eval()


class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        target_image = io.imread(self.image_paths[idx])
        gray_image = io.imread(self.image_paths[idx], as_gray=True) * 255

        target_image = skimage.transform.resize(target_image, (256, 256))
        gray_image = skimage.transform.resize(gray_image, (32, 32))

        gray_image = PIL.Image.fromarray(gray_image)

        lab_image = transforms.ToTensor()(rgb_to_lab(target_image))
        lab_image = lab_image.unsqueeze(0).to(device)

        target_image = skimage.transform.resize(target_image, (32, 32))

        with torch.no_grad():
            pil_image = preprocess(gray_image).unsqueeze(0).to(device)

            prefix = clip_model.encode_image(pil_image).to(device, dtype=torch.float32)
            prefix_embed = captioning_model.clip_project(prefix).reshape(1, prefix_length, -1)
            token_embeddings = get_token_embedings(captioning_model, tokenizer, embed=prefix_embed).to(device)
            token_embeddings = transforms.Resize((12, 192))(token_embeddings)
            token_embeddings = token_embeddings.view(1, 12 * 192)

            min_token_embeddings = torch.min(token_embeddings)
            max_token_embeddings = torch.max(token_embeddings)
            token_embeddings = (token_embeddings - min_token_embeddings) / (max_token_embeddings - min_token_embeddings)

            generated_image = colorization_model(lab_image)
            generated_image = transforms.Resize((32, 32))(generated_image)
            generated_image = generated_image.view(1, 3 * 32 * 32)

            combined_features = torch.cat((token_embeddings, generated_image), 1).float().to(device)

        target_image = torch.from_numpy(target_image).float().to(device)

        return combined_features, target_image


def get_token_embedings(model, tokenizer, embed):
    token_embeddings = []
    tokens = None
    prompt = None
    entry_count = 1
    entry_length = 12  # Max number of words per caption
    top_p = 0.8
    temperature = 1.
    stop_token_index = tokenizer.encode('.')[0]
    filter_value = -float("Inf")

    with torch.no_grad():
        for _ in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens).to(device)
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits.to(device)
                logits = (logits[:, -1, :] / (temperature if temperature > 0 else 1.0)).to(device)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1).to(device)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0).to(device)
                next_token_embed = model.gpt.transformer.wte(next_token).to(device)
                token_embeddings.append(next_token_embed.cpu())

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)

                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

    zeros = torch.zeros(1, 12, 768)
    for i in range(len(token_embeddings)):
        zeros[0][i] = token_embeddings[i]

    return zeros


def rgb_to_lab(rgb):
    """ Transforms a PIL RGB image into a Lab tensor """
    img = rgb2lab(rgb).astype('float32')
    lab_img = (img[..., 0:1] / 50.) - 1.
    return lab_img


def main():
    n_epochs = 100
    batch_size = 64

    # Loading data
    training_dir = 'data/train2014/'
    val_dir = 'data/val2014/'

    train_set = ImageDataset(training_dir)
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = ImageDataset(val_dir)
    val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    input_size = 5376  # the size of the combined embedding
    hidden_size = 3072  # the number of the final image
    output_size = 3072  # the number of the final image

    combined_model = CombinedColorizationModel(input_size, hidden_size, output_size)
    combined_model = combined_model.to(device)
    combined_model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Minimum validation loss
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        clear_output()
        print(f">>> Training Epoch {epoch + 1}")
        clear_output()
        sys.stdout.flush()
        progress = tqdm(total=len(train_set))
        # Training
        for features, target_images in train_set:
            # Forward pass
            optimizer.zero_grad()
            colored_image = combined_model(features, len(features)).to(device)
            loss = criterion(colored_image, target_images.to(device)).to(device)

            # Back propagation
            loss.backward()
            optimizer.step()

            progress.set_postfix({"Training Loss": loss.item()})
            progress.update()
        progress.close()
        # Validation
        val_loss = 0
        clear_output()
        print(f">>> Validation Epoch {epoch + 1}")
        clear_output()
        sys.stdout.flush()
        progress = tqdm(total=len(val_set))
        for features, target_images in val_set:
            with torch.no_grad():
                colored_image = combined_model(features, len(features)).to(device)
                loss = criterion(colored_image.to(device), target_images.to(device)).to(device)
                val_loss += loss.item()
                progress.set_postfix({"Validation Loss": loss.item()})
                progress.update()
        progress.close()
        # If the model achieved a validation loss lower than the previous best one, the current model parameters are
        # saved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(combined_model.state_dict(), 'checkpoints_combined_model/cp-{}.pt'.format(epoch))


if __name__ == '__main__':
    main()
