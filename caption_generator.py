import clip
import os
import torch
import torch.nn.functional as nnf
from transformers import GPT2Tokenizer
import skimage.io as io
import PIL.Image
import matplotlib.pyplot as plt
from captioning_model import ClipCaptionModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def generate_caption(model, tokenizer, embed):
    generated_list = []
    stop_token_index = tokenizer.encode('.')[0]
    filter_value = -float("Inf")
    entry_count = 1
    entry_length = 67  # Max caption length
    top_p = 0.8
    temperature = 1
    tokens = None

    with torch.no_grad():
        for _ in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode())
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)

                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main():
    prefix_length = 10
    model_path = os.path.join('checkpoints_captioning/coco_prefix_latest.pt')

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = ClipCaptionModel(prefix_length)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model = model.eval()
    model = model.to(device)

    for image in os.listdir("data/test2014/"):
        image = io.imread("data/test2014/" + image, as_gray=True) * 255

        pil_image = PIL.Image.fromarray(image)

        image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            generated_text_prefix = generate_caption(model, tokenizer, embed=prefix_embed)

        plt.imshow(pil_image)
        plt.xlabel(generated_text_prefix)
        plt.show()


if __name__ == '__main__':
    main()
