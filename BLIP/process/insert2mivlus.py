import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_path = '/data/wjy/blip-image-captioning-large'  # 你可以根据需要替换成其他预训练模型

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to("cuda")

img_url = '0001.jpg'
raw_image = Image.open(img_url).convert('RGB')

inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.vision_model(**inputs).last_hidden_state
output_vector = out.detach().cpu().numpy().tolist()
print(output_vector[0])
print(len(output_vector))
print(out.shape)
