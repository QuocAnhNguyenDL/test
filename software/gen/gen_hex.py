import os
import cv2

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

def rgb_to_hex(r, g, b):
    """Chuyển đổi giá trị R, G, B thành dạng hex."""
    return f'{r:02x}{g:02x}{b:02x}'

def convert_images_to_txt(input_folder, output_folder):
    """Chuyển đổi tất cả ảnh trong thư mục thành file txt và lưu vào cấu trúc thư mục tương tự."""
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Đọc ảnh
                img_path = "/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/23.jpg"
                img = cv2.imread(img_path)
                
                # Kiểm tra ảnh có đúng kích thước 128x128 không
                if img.shape[:2] != (128, 128):
                    print(f"Bỏ qua ảnh {file} do không đúng kích thước 128x128.")
                    continue
                
                # Chuyển BGR sang RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Chuyển từng pixel thành hex
                
                hex_values = []
                for row in img_rgb:
                    for pixel in row:
                        r, g, b = pixel
                        hex_values.append(rgb_to_hex(r, g, b))
                
                # Tạo đường dẫn tương ứng cho thư mục đầu ra
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Lưu file txt
                txt_filename = os.path.splitext(file)[0] + ".txt"
                txt_path = os.path.join(output_dir, txt_filename)
                
                with open(txt_path, 'w') as f:
                    f.write("\n".join(hex_values))
                
                print(f"Đã xử lý: {file} -> {txt_path}")

# Thư mục đầu vào và đầu ra
input_root = "/home/quocna/project/DOAN2/dataset/datasettotrain"  # Thay bằng thư mục chứa train, val, test
output_root = "/home/quocna/project/DOAN2/dataset/dataset"  # Thư mục sẽ chứa các file txt

# Chạy chuyển đổi
convert_images_to_txt(input_root, output_root)
