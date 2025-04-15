# import os
# import shutil
#
#
# def rename_and_copy_images(src_folder, dst_folder):
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#
#     images = sorted([f for f in os.listdir(src_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
#
#     for img in images:
#         src_path = os.path.join(src_folder, img)
#
#         # 检查是否需要重命名
#         if img.endswith('.jpg') and '0050' <= img[:4] <= '0095':
#             new_index = f"{int(img[:4]) - 1:04d}"
#             new_name = new_index + img[4:]
#         else:
#             new_name = img  # 其他图片保持原名
#
#         dst_path = os.path.join(dst_folder, new_name)
#         shutil.copy2(src_path, dst_path)
#         print(f"Copied: {src_path} -> {dst_path}")
#
#
# src_folder = r"D:\maskrcnn\datasets\新数据（首诊、复诊）\复诊"
# dst_folder = r"D:\maskrcnn\datasets\新数据（首诊、复诊）\复诊_重命名"
# rename_and_copy_images(src_folder, dst_folder)


import os
import shutil

# 源文件夹（包含 JPG 图片）
source_folder = r"D:\maskrcnn\datasets\images"
# 目标文件夹
target_folder = r"D:\maskrcnn\datasets\新数据（首诊、复诊）\首诊_first"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有 .jpg 文件
for filename in os.listdir(source_folder):
    if filename.lower().endswith(".jpg"):  # 只处理 JPG 图片
        old_path = os.path.join(source_folder, filename)
        new_path = os.path.join(target_folder, filename)

        # 复制文件
        shutil.copy2(old_path, new_path)
        print(f"Copied: {filename}")

print("所有 JPG 文件已复制到目标文件夹。")

