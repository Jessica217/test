# from PIL import Image
# # from ..datasets import images
# # 打开图像文件
# image = Image.open("0001.jpg")  # 300 *299
#
# # 获取图像尺寸 (宽度, 高度)
# width, height = image.size
#
#
# print(f"Image size: {width}x{height}")
#  ##TEST 0001.JPG   582*583

from PIL import Image

# 打开图像文件
image_path = "extra_50_0002.jpg"
image = Image.open(image_path)

# 将图像缩放到 300x300
image_resized = image.resize((300, 300))

# 保存缩放后的图像
resized_image_path = "your_resized_image_path.jpg"
image_resized.save(resized_image_path)

# 显示缩放后的图像（可选）
# image_resized.show()

print(f"Image resized to: 300x300")
