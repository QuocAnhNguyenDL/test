import os
import random

# Thư mục gốc
base_dir = "/home/quocna/project/DOAN2/dataset/dataset/test"
output_file = "/home/quocna/project/DOAN2/fire_detec/software/txt/dataset.txt"
shuffled_output_file = "/home/quocna/project/DOAN2/fire_detec/software/txt/dataset.txt"

# Duyệt qua tất cả các tệp trong thư mục 'test'
file_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Tạo đường dẫn đầy đủ
        full_path = os.path.join(root, file)
        file_paths.append(full_path)

# Xáo trộn các đường dẫn
random.shuffle(file_paths)

# Ghi các đường dẫn xáo trộn vào tệp output
with open(shuffled_output_file, "w") as f:
    for path in file_paths:
        f.write(path + "\n")

print(f"Đã lưu các đường dẫn xáo trộn vào {shuffled_output_file}")
