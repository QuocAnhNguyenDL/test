import numpy as np
import torch
from torchinfo import summary
from torchvision import transforms
from PIL import Image
from torch.quantization.observer import MovingAverageMinMaxObserver

from model.model import *



# Hàm dự đoán
def predict_image(image_path, model):
    # Định nghĩa các phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())  # PIL -> NumPy -> Tensor, giữ nguyên giá trị
    ])

    # Đọc ảnh
    image = Image.open(image_path).convert('RGB')
    
    # Áp dụng phép biến đổi
    image_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch

    # Dự đoán
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.squeeze().tolist()

def get_intermediate_output(image_path, model, layer_name="conv1"):
    """
    Trích xuất kết quả trung gian từ một lớp cụ thể.
    Args:
        image_path (str): Đường dẫn tới ảnh đầu vào.
        model (torch.nn.Module): Mô hình PyTorch.
        layer_name (str): Tên lớp cần lấy kết quả trung gian.
    Returns:
        torch.Tensor: Kết quả trung gian từ lớp.
    """
    # Biến đổi ảnh
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())  # PIL -> NumPy -> Tensor, giữ nguyên giá trị
    ])

    # Đọc và xử lý ảnh
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch

    # Hook để lấy kết quả trung gian
    intermediate_output = None

    def hook_fn(module, input, output):
        nonlocal intermediate_output
        print(input[0].shape, output.shape)
        print(f"Hook activated for layer: {module}")

        # Trích xuất các giá trị
        pre_bias = input[0].detach()  # Trước khi cộng bias
        post_bias = output.detach()  # Sau khi cộng bias
        activated = torch.sigmoid(post_bias)  # Hàm kích hoạt `tanh` (Sigmoid là tên nhầm, chính xác là Tanh)

        # Lưu trữ kết quả
        intermediate_output = {
            "pre_bias": pre_bias,
            "post_bias": post_bias,
            "activated": activated,
        }

    # Đăng ký hook vào lớp
    found_layer = False
    for name, layer in model.net.named_modules():
        print(f"Checking layer: {name}")  # In tên lớp để kiểm tra
        if name == layer_name:
            layer.register_forward_hook(hook_fn)
            found_layer = True

    if not found_layer:
        raise ValueError(f"Layer '{layer_name}' not found in model!")

    # Chạy dự đoán để kích hoạt hook
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor)

    if intermediate_output is None:
        raise RuntimeError("Hook did not capture any output. Verify the layer is activated in forward pass.")

    return intermediate_output

def save_pixel_values(output_dict, output_file, height, width, channels):
    """
    Lưu các giá trị pixel (x, y, channel) kèm các kết quả: trước bias, sau bias, sau kích hoạt.
    Args:
        output_dict (dict): Kết quả đầu ra từ hook (gồm pre_bias, post_bias, activated).
        output_file (str): Đường dẫn file để lưu kết quả.
        height (int): Chiều cao của ảnh.
        width (int): Chiều rộng của ảnh.
        channels (int): Số lượng kênh.
    """
    pre_bias = output_dict["pre_bias"]
    post_bias = output_dict["post_bias"]
    activated = output_dict["activated"]

    with open(output_file, "w") as f:
        for c in range(channels):  # Duyệt từng kênh
            for x in range(height):  # Duyệt từng hàng
                for y in range(width):  # Duyệt từng cột
                    # Lấy giá trị tại vị trí (x, y, c)
                    if c<=2:
                        pre_bias_val = pre_bias[0, c, x, y].item()
                    else:
                        pre_bias_val = -1
                    post_bias_val = post_bias[0, c, x, y].item()
                    activated_val = activated[0, c, x, y].item()

                    # Lưu vào file
                    f.write(f"{x} {y} {c} {pre_bias_val:.6f} {post_bias_val:.6f} {activated_val:.6f}\n")

    print(f"Kết quả đã được lưu tại: {output_file}")

if __name__ == '__main__':

    model = Model()
    image_path = '/home/quocna/project/DOAN2/dataset/datasettotrain/test/Fire/64.jpg'

    model_bth = Model()
    # print(hasattr(model, 'fuse_model'))
    model_bth.eval()

    model.eval()
    model.qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
        weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
    )

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection_quant.pth'))
    summary(model)
    model_bth.load_state_dict(torch.load('/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection.pth', map_location='cpu'))
    summary(model_bth)

    conv1_output = get_intermediate_output(image_path, model, layer_name="conv1")
    dequantized_output = {
        "pre_bias": conv1_output["pre_bias"].dequantize(),
        "post_bias": conv1_output["post_bias"],
        "activated": conv1_output["activated"].dequantize(),
    }

    # Kết quả có kích thước [1, 16, 128, 128]
    #print(f"conv1 output shape: {conv1_output.shape}")

    # Xuất kết quả ra file .npy
    output_file = "/home/quocna/project/DOAN2/fire_detec/software/output_from_model/conv1/conv1_with_positions.txt"
    save_pixel_values(dequantized_output, output_file, 128, 128,16)
    # output_file = "/home/quocna/project/DOAN2/fire_detec/software/output_from_model/conv1/conv1_dequant.txt"
    # np.savetxt(output_file, dequantized_output.reshape(-1), fmt="%.6f", delimiter=",")
    # output_file = "/home/quocna/project/DOAN2/fire_detec/software/output_from_model/conv1/conv1_quant.txt"
    # np.savetxt(output_file, dequantized_output.reshape(-1)/4.234874725341797, fmt="%.6f", delimiter=",")
    # print(f"Kết quả đã được lưu tại: {output_file}")


    # Gọi hàm dự đoán
    predicted_class, probabilities = predict_image(image_path, model)

    # Hiển thị kết quả
    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {probabilities}")
