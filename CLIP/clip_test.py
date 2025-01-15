import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("../BLIP/0001.jpg")).unsqueeze(0).to(device) #输入
text = clip.tokenize(["two kidneys","a picture with acute pylenerphritis","a picture with urinary tract infection"]).to(device) # 向量数据库

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print("image_features:", image_features)
    print("text_features:", text_features)

    logits_per_image, logits_per_text = model(image, text)
    print("logits_per_image:", logits_per_image)
    print("logits_per_text:", logits_per_text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

