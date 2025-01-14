import cv2

# # Đọc ảnh
# image = cv2.imread("/home/quocna/project/backup/fire_detection/data/fire_RGB11/1.jpg")

# # Kiểm tra kích thước ảnh
# height, width, channels = image.shape  # shape: (height, width, channels)
# total_pixels = height * width

# print(f"So kenh màu: {channels}")
# print(f"Chieu cao: {height} pixel")
# print(f"Chieu rộng: {width} pixel")
# print(f"Tong so pixel: {total_pixels}")
#/home/quocna/project/DOAN2/dataset/dataset/fire/matrix/1095.txt

# Mở file và đọc nội dung
# file_path = "/home/quocna/project/DOAN2/dataset/dataset/fire/matrix/1095.txt"  # Đường dẫn tới file
# with open(file_path, "r") as file:
#     content = file.read()  # Đọc toàn bộ file

# # Tách các giá trị
# values = content.strip().split(",")  # Tách các giá trị dựa trên dấu phẩy

# # Đếm số lượng các giá trị
# count = len(values)

# # In từng giá trị và tổng số lượng
# for value in values:
#     print(value)

# print(f"\nTổng số lượng các giá trị: {count}")

# import numpy as np
# import torch

# def process_matrix(file_path, size=(240, 240)):
#     """
#     Đọc file matrix (dạng 1 hàng), làm tròn và reshape thành ảnh kích thước 240x240.
#     Args:
#         file_path (str): Đường dẫn đến file matrix.
#         size (tuple): Kích thước mong muốn của ảnh (240, 240).
#     Returns:
#         torch.Tensor: Tensor có kích thước [1, 240, 240].
#     """
#     # Đọc file và tách các giá trị
#     with open(file_path, 'r') as f:
#         data = f.read().split(',')

#     # Chuyển dữ liệu sang dạng float và làm tròn
#     matrix = np.round(np.array(data, dtype=np.float32))

#     # Reshape thành ma trận vuông
#     matrix_resized = matrix.reshape(size)

#     # Thêm 1 kênh và chuyển thành tensor
#     matrix_tensor = torch.tensor(matrix_resized, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
#     return matrix_tensor

# # Thử nghiệm
# file_path = '/home/quocna/project/DOAN2/dataset/dataset/fire/matrix/108.txt'  
# matrix_tensor = process_matrix('/home/quocna/Downloads/test.png')
# print(f"Matrix tensor shape: {matrix_tensor.shape}")
# print(matrix_tensor)

# import cv2


# path = "/home/quocna/project/DOAN2/dataset/dataset/nofire/rgb/359.jpg"
# img = cv2.imread(path)

# cv2.imshow("name", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2

# # Đọc file txt và chuyển về numpy array
# def txt_to_image(file_path, width=240, height=240, threshold=40):
#     # Đọc file txt
#     with open(file_path, 'r') as file:
#         data = file.read()
    
#     # Chuyển đổi dữ liệu từ chuỗi về float và tạo numpy array
#     pixel_values = np.array([float(x) for x in data.split(',')])
    
#     # Reshape thành ma trận ảnh (240x240)
#     image = pixel_values.reshape((height, width))
    
#     # Áp dụng điều kiện màu: > 50 là trắng, <= 50 là đen
#     binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    
#     return binary_image

# # Đường dẫn file txt
# file_path = '/home/quocna/project/DOAN2/dataset/dataset/nofire/matrix/52.txt'  # Thay thế đường dẫn của bạn

# # Chuyển đổi và hiển thị ảnh
# binary_image = txt_to_image(file_path)

# # Hiển thị ảnh
# cv2.imshow('Binary Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2

# def hex_to_image_with_mask(file_path, width=240, height=240):
#     # Đọc file txt và làm sạch dữ liệu
#     with open(file_path, 'r') as file:
#         hex_data = file.read().replace('\n', ',').split(',')

#     # Loại bỏ các phần tử rỗng hoặc không hợp lệ
#     hex_data = [x.strip() for x in hex_data if x.strip() != '']

#     # Kiểm tra số lượng pixel
#     if len(hex_data) != width * height:
#         raise ValueError(f"Số lượng pixel không hợp lệ: {len(hex_data)} (yêu cầu {width * height})")

#     # Tạo mảng numpy để chứa ảnh (height, width, 3) - 3 channels RGB
#     image = np.zeros((height, width, 3), dtype=np.uint8)

#     # Duyệt qua từng pixel và giải mã giá trị hex
#     for i, hex_value in enumerate(hex_data):
#         # Chuyển hex thành giá trị integer
#         hex_int = int(hex_value, 16)
#         r = (hex_int >> 24) & 0xFF  # Lấy 8 bit cao nhất (R)
#         g = (hex_int >> 16) & 0xFF  # Tiếp theo 8 bit (G)
#         b = (hex_int >> 8) & 0xFF   # Tiếp theo 8 bit (B)
#         mask = hex_int & 0xFF       # Mask nằm ở 8 bit cuối cùng

#         # Áp dụng mask (2 bit cuối)
#         mask_bits = mask & 0b11
#         if mask_bits == 0b01:  # Làm sáng pixel
#             r = min(r + 50, 255)
#             g = min(g + 50, 255)
#             b = min(b + 50, 255)
#         elif mask_bits == 0b10:  # Làm tối pixel
#             r = max(r - 50, 0)
#             g = max(g - 50, 0)
#             b = max(b - 50, 0)
#         elif mask_bits == 0b11:  # Chuyển pixel thành màu đen
#             r, g, b = 0, 0, 0

#         # Tính toán vị trí pixel trên ảnh
#         row = i // width
#         col = i % width

#         # Gán giá trị R, G, B vào ảnh
#         image[row, col] = [b, g, r]  # OpenCV dùng BGR, nên thứ tự là B, G, R

#     return image

# Đường dẫn file txt
# file_path = '/home/quocna/project/DOAN2/dataset/dataset/fire/datatrain/601.txt'  # Thay bằng đường dẫn file txt của bạn

# # Chuyển đổi và hiển thị ảnh
# image = hex_to_image_with_mask(file_path)

# # Hiển thị ảnh
# cv2.imshow('Image with Mask', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Đọc file txt và chuyển giá trị hex RGB thành ảnh
# def hex_txt_to_image(txt_file, output_image):
#     # Tạo một mảng rỗng với kích thước 128x128x3
#     image = np.zeros((128, 128, 3), dtype=np.uint8)

#     # Mở file và đọc từng dòng
#     with open(txt_file, 'r') as file:
#         lines = file.readlines()
    
#     # Kiểm tra đủ 128x128 giá trị
#     if len(lines) != 128 * 128:
#         raise ValueError("File không chứa đủ 128x128 giá trị.")

#     # Duyệt qua từng pixel và gán giá trị màu
#     idx = 0
#     for i in range(128):
#         for j in range(128):
#             # Loại bỏ ký tự xuống dòng
#             hex_color = lines[idx].strip()
            
#             # Chuyển đổi hex sang giá trị R, G, B
#             r = int(hex_color[0:2], 16)
#             g = int(hex_color[2:4], 16)
#             b = int(hex_color[4:6], 16)
            
#             # Gán giá trị cho pixel (Lưu ý OpenCV sử dụng BGR thay vì RGB)
#             image[i, j] = [b, g, r]
#             idx += 1

#     # Lưu ảnh kết quả
#     cv2.imwrite(output_image, image)
#     print(f"Ảnh đã được lưu tại: {output_image}")
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchinfo import summary
# from torchvision import transforms
# from PIL import Image

# # Khởi tạo mô hình
# import math
# import torch
# import torch.nn as nn
# from model.model import *
# from fxpmath import Fxp
# from torch.quantization.observer import MovingAverageMinMaxObserver

# # Hàm dự đoán
# def predict_image(image_path, model):
#     # Định nghĩa phép biến đổi ảnh không chuẩn hóa
#     def custom_transform(image):
#         image_array = np.array(image, dtype=np.float32)  # Giữ giá trị trong dải [0, 255]
#         image_tensor = torch.from_numpy(image_array.transpose((2, 0, 1)))  # Chuyển sang Tensor (C, H, W)
#         return image_tensor.unsqueeze(0)  # Thêm chiều batch

#     # Đọc ảnh
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = custom_transform(image)  # Không chuẩn hóa giá trị pixel

#     # Dự đoán
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probabilities = outputs
#         predicted_class = torch.argmax(probabilities, dim=1).item()

#     return predicted_class, probabilities.squeeze().tolist()

# def get_intermediate_output(image_path, model, layer_name="conv1"):
#     """
#     Trích xuất kết quả trung gian từ một lớp cụ thể.
#     Args:
#         image_path (str): Đường dẫn tới ảnh đầu vào.
#         model (torch.nn.Module): Mô hình PyTorch.
#         layer_name (str): Tên lớp cần lấy kết quả trung gian.
#     Returns:
#         torch.Tensor: Kết quả trung gian từ lớp.
#     """
#     def custom_transform(image):
#         image_array = np.array(image, dtype=np.float32)  # Giữ giá trị trong dải [0, 255]
#         image_tensor = torch.from_numpy(image_array.transpose((2, 0, 1)))  # Chuyển sang Tensor (C, H, W)
#         return image_tensor.unsqueeze(0)  # Thêm chiều batch

#     # Đọc và xử lý ảnh
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = custom_transform(image)

#     # Hook để lấy kết quả trung gian
#     intermediate_output = None

#     def hook_fn(module, input, output):
#         nonlocal intermediate_output
#         print(f"Hook activated for layer: {module}")
#         intermediate_output = output

#     # Đăng ký hook vào lớp
#     found_layer = False
#     for name, layer in model.net.named_modules():
#         print(f"Checking layer: {name}")  # In tên lớp để kiểm tra
#         if name == layer_name:
#             layer.register_forward_hook(hook_fn)
#             found_layer = True

#     if not found_layer:
#         raise ValueError(f"Layer '{layer_name}' not found in model!")

#     # Chạy dự đoán để kích hoạt hook
#     model.eval()
#     with torch.no_grad():
#         _ = model(image_tensor)

#     if intermediate_output is None:
#         raise RuntimeError("Hook did not capture any output. Verify the layer is activated in forward pass.")

#     return intermediate_output

# if __name__ == '__main__':
#     model = Model()
#     image_path = '/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/3.jpg'

#     model_bth = Model()
#     model_bth.eval()
#     model_bth.fuse_model()

#     model.eval()
#     model.fuse_model()
#     model.qconfig = torch.quantization.QConfig(
#         activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
#         weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
#     )

#     torch.quantization.prepare(model, inplace=True)
#     torch.quantization.convert(model, inplace=True)

#     model.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection_quant.pth'))
#     summary(model)
#     model_bth.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection.pth', map_location='cpu'))
#     summary(model_bth)

#     conv1_output = get_intermediate_output(image_path, model, layer_name="fc1")
#     dequantized_output = conv1_output.dequantize()

#     # Kết quả có kích thước [1, 16, 128, 128]
#     print(f"conv1 output shape: {conv1_output.shape}")

#     # Xuất kết quả ra file .npy
#     output_file = "/home/quocna/project/DOAN2/fire_detec/software/output_from_model/conv1/conv1_dequant.txt"
#     np.savetxt(output_file, dequantized_output.reshape(-1), fmt="%.6f", delimiter=",")
#     print(f"Kết quả đã được lưu tại: {output_file}")

#     # Gọi hàm dự đoán
#     predicted_class, probabilities = predict_image(image_path, model_bth)

#     # Hiển thị kết quả
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Probabilities: {probabilities}")
# import cv2
# import numpy as np

# # Đường dẫn tới ảnh
# image_path = "/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/64.jpg"

# # Đọc ảnh
# image = cv2.imread(image_path)

# # Kiểm tra nếu ảnh không được đọc thành công
# if image is None:
#     print("Không thể đọc ảnh từ đường dẫn:", image_path)
#     exit()

# # Hiển thị thông tin về kênh màu
# print("Kích thước ảnh:", image.shape)
# print("Kênh màu tại pixel (0,0):", image[0, 0])

# # Hàm để in giá trị của các kênh màu tại một vị trí dưới dạng HEX
# def show_pixel_value(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Nếu bấm chuột trái
#         b, g, r = image[y, x]
#         hex_value = f"#{r:02x}{g:02x}{b:02x}"  # Chuyển đổi RGB sang HEX
#         print(f"Pixel tại ({x}, {y}): {hex_value}")


# # Tạo cửa sổ và liên kết sự kiện chuột
# cv2.namedWindow("Image")
# cv2.setMouseCallback("Image", show_pixel_value)

# # Hiển thị ảnh
# while True:
#     cv2.imshow("Image", image)
#     key = cv2.waitKey(1)
#     if key == 27:  # Bấm phím ESC để thoát
#         break

# # Đóng tất cả cửa sổ
# cv2.destroyAllWindows()

# import torch
# import torch.nn.functional as F
# from torchvision.transforms import ToTensor
# import numpy as np
# import torch
# from torchinfo import summary
# from torchvision import transforms
# from PIL import Image
# from torch.quantization.observer import MovingAverageMinMaxObserver

# from model.model import *

# model = Model()
# image_path = '/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/64.jpg'

# model_bth = Model()
# # print(hasattr(model, 'fuse_model'))
# model_bth.eval()

# model.eval()
# model.qconfig = torch.quantization.QConfig(
#     activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
#     weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
# )

# transform = transforms.Compose([
#     transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())  # PIL -> NumPy -> Tensor, giữ nguyên giá trị
# ])

# torch.quantization.prepare(model, inplace=True)
# torch.quantization.convert(model, inplace=True)

# model.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection_quant.pth'))
# summary(model)
# model_bth.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection.pth', map_location='cpu'))
# summary(model_bth)

# # Tải ảnh và tiền xử lý
# image_path = '/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/64.jpg'
# image = Image.open(image_path).convert('RGB')
# # Áp dụng phép biến đổi
# image_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch

# # Duyệt qua từng bước trong tầng đầu tiên
# conv1 = model.net.conv1
# sigmoid = model.net.tanh
# pool1 = model.net.pool1

# # Tích chập
# conv1_output = conv1(image_tensor)
# print("Conv1 Output Shape:", conv1_output.shape)
# print("Conv1 Output Pixel [0, 0, 0]:", conv1_output[0, 0, 0, 0].item())  # Pixel đầu tiên của kênh 0

# # Kích hoạt
# sigmoid_output = sigmoid(conv1_output)
# print("Sigmoid Output Pixel [0, 0, 0]:", sigmoid_output[0, 0, 0, 0].item())

# # Pooling
# pool1_output = pool1(sigmoid_output)
# print("Pool1 Output Shape:", pool1_output.shape)
# print("Pool1 Output Pixel [0, 0, 0]:", pool1_output[0, 0, 0, 0].item())
# import os
# import random

# def generate_random_hex():
#     """Tạo giá trị ngẫu nhiên dạng hex với 6 ký tự."""
#     return ''.join(random.choices('0123456789abcdef', k=6))

# def process_file(file_path):
#     """Thay đổi 5 dòng đầu và 5 dòng cuối của file thành giá trị ngẫu nhiên."""
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     # Đảm bảo file có ít nhất 10 dòng để thay đổi
#     if len(lines) <= 10:
#         print(f"File {file_path} có ít hơn hoặc bằng 10 dòng, bỏ qua.")
#         return

#     # Thay đổi 5 dòng đầu
#     for i in range(5):
#         lines[i] = generate_random_hex() + '\n'

#     # Thay đổi 5 dòng cuối
#     for i in range(-5, 0):
#         lines[i] = generate_random_hex() + '\n'

#     # Ghi lại nội dung mới vào file
#     with open(file_path, 'w') as file:
#         file.writelines(lines)

#     print(f"Đã xử lý file: {file_path}")

# def process_files_in_directory(directory):
#     """Xử lý tất cả các file .txt trong thư mục và các thư mục con."""
#     for root, dirs, files in os.walk(directory):  # Sử dụng os.walk để duyệt qua tất cả thư mục con
#         for filename in files:
#             if filename.endswith('.txt'):  # Kiểm tra nếu là file .txt
#                 file_path = os.path.join(root, filename)  # Đường dẫn đầy đủ tới file
#                 process_file(file_path)  # Gọi hàm xử lý file
#                 print(f"Đã xử lý file: {file_path}")

# # Thay đổi đường dẫn tới thư mục chứa các file txt của bạn
# directory_path = '/home/quocna/project/DOAN2/dataset/dataset/test/Fire'
# process_files_in_directory(directory_path)
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Giả sử bạn đã có:
# 1. Mô hình đã được huấn luyện: model
# 2. Tập dữ liệu kiểm tra: X_test, y_test

# def evaluate_model_accuracy(model, X_test, y_test):
#     """
#     Đánh giá độ chính xác của mô hình trên tập dữ liệu kiểm tra.

#     Args:
#         model: Mô hình đã được huấn luyện.
#         X_test: Tập dữ liệu đầu vào để kiểm tra.
#         y_test: Nhãn thực tế của tập dữ liệu kiểm tra.

#     Returns:
#         accuracy: Độ chính xác của mô hình trên tập kiểm tra.
#     """
#     try:
#         # Dự đoán nhãn từ tập X_test
#         y_pred = model.predict(X_test)

#         # Tính độ chính xác
#         accuracy = accuracy_score(y_test, y_pred)

#         print(f"Accuracy of the model on test data: {accuracy * 100:.2f}%")
#         return accuracy

#     except Exception as e:
#         print(f"An error occurred during model evaluation: {e}")
#         return None

# # Ví dụ cách sử dụng
# if __name__ == "__main__":
#     # Giả lập dữ liệu (nếu bạn không có dữ liệu sẵn)
#     from sklearn.datasets import make_classification
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import train_test_split

#     # Tạo tập dữ liệu mẫu
#     X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Huấn luyện mô hình mẫu
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Đánh giá mô hình
#     evaluate_model_accuracy(model, X_test, y_test)
