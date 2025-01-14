import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.quantization.observer import MovingAverageMinMaxObserver


from model.model import *

# Đường dẫn đến mô hình và ảnh
#image_path = "/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/50.jpg"  # Thay bằng đường dẫn tới ảnh cần kiểm tra

# Load mô hình
model = Model()
model.eval()
model.qconfig = torch.quantization.QConfig(
    activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
    weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
)

torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

model.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection_quant.pth'))

# Định nghĩa các phép biến đổi ảnh
transform = transforms.Compose([
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())  # PIL -> NumPy -> Tensor, giữ nguyên giá trị
])
criterion = nn.CrossEntropyLoss()

image_dir = "/home/quocna/project/DOAN2/dataset/datasettotrain/test/no-Fire"
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    # Load ảnh
    image = Image.open(image_path).convert("RGB")  # Đảm bảo ảnh có 3 kênh
    image = transform(image)  # Áp dụng các phép biến đổi
    image = image.unsqueeze(0)  # Thêm batch dimension

    with torch.no_grad():
        output = model(image)  # Chạy mô hình với input là ảnh
        #print(output)
        probabilities = output  # Chuyển đổi thành xác suất
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Lấy lớp có xác suất cao nhất
        class_probabilities = probabilities.squeeze().tolist()  # Chuyển đổi xác suất thành danh sách

    # In kết quả

    print(predicted_class)
    #print(f"Dự đoán của mô hình: Lớp {predicted_class}")
    #print("Xác suất cho từng lớp:")
    #for i, prob in enumerate(class_probabilities):
        #print(f"  Lớp {i}: {prob * 100:.2f}%")