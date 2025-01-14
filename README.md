# ĐỒ ÁN 2
---

## Thông tin chung
- Tên   : Nguyễn Anh Quốc 
- MSSV  : 21522526 
- Lớp   : MTCL2021 
- Email : 21522526@uit.edu.vn 
- Đồ án : 
- GVHD  : Trương Văn Cương 
---

## Mục lục
1. [RISC-V](#RISC-v)
2. [GIAO THỨC AXI](#GIAO-THỨC-AXI)
3. [KIẾN TRÚC DỰ KIẾN](#KIẾN-TRÚC-DỰ-KIẾN)
4. [KIẾN TRÚC MOBILENETv2](#KIẾN-TRÚC-MOBILENETv2)
5. [TÀI LIỆU THAM KHẢO](#TÀI-LIỆU-THAM-KHẢO)
---

## RISC-V
RISC-V là một kiến trúc tập lệnh (Instruction Set Architecture - ISA) theo kiểu RISC (Reduced Instruction Set Computer), được thiết kế theo hướng mở và miễn phí bản quyền. Một trong những mục tiêu chính của RISC-V là cung cấp một kiến trúc ISA có khả năng mở rộng, linh hoạt, và không bị ràng buộc bởi chi phí bản quyền, giúp nó trở thành nền tảng lý tưởng cho nhiều loại thiết bị và hệ thống, từ các bộ vi xử lý đơn giản trong thiết bị IoT đến các siêu máy tính hiệu năng cao.

RISC-V thiết kế trong dự án để chạy các lệnh trong Instruction set 32I (RV32I).

<p align="center">
  <img src="https://github.com/user-attachments/assets/716bcb33-24e6-4cd3-bc10-005d176534cd" alt="Ảnh mẫu" />
</p>

Sơ đồ của RISC-V:![image](https://github.com/user-attachments/assets/7f8d099d-3f21-4998-8743-05c6038fcaf5)

## GIAO THỨC AXI
SoC là một vi mạch được tích hợp các thành phần của một máy tính hoặc các hệ thống điện tử khác. Các thành phần này được chia làm Master và các Slave. Sự giao tiếp giữa các thành phần được thực hiện thông qua BUS. Chức năng chính của BUS bao gồm: 
- Liên kết các thành phần trong hệ thống.
- Phân xử truy cập.
- Giải mã đia chỉ.

AXI là một giao thức BUS trong họ AMBA (Advanced Microcontroller Bus Architecure) được phát triển bởi ARM. Giao thức AXI quy định chuẩn giao tiếp giữa một Master - một BUS, một Slave - một BUS hoặc giữa một Master và một Slave. Cấu trúc phân kênh của AXI gôm 5 loại kênh "độc lâp" - tức là mỗi kênh có một nhiệm vụ khác nhau và không bị phụ thuộc vào kênh khác. 5 kênh này bao gồm:
- Kênh địa chỉ đọc
- Kênh dữ liệu đọc
- Kênh địa chỉ ghi
- Kênh dữ liệu ghi
- Kênh đáp ứng ghi

## GIỚI THIỆU VỀ DEPTHWISE SEPARABLE CONVOLUTION 
- Đây là một kỹ thuật nổi tiếp được áp dụng khi muốn xây dựng một mạng CNN trên các thiết bị mà bị giới hạn về phần cứng
- Được áp dụng trong những mô hình CNN nổi tiếng như Shufflenet, Mobilenet, ...
### Standard Convolutions ( SC)
![image](https://github.com/user-attachments/assets/56ba6665-c5cd-4d52-830d-5fcbac694938)

Giả sử đầu vào là 1 feature map có kích thược 5@10x10 và bộ lọc có kích thước là 5@3x3. Khi ta thực hiện phps tích chập trên toàn bộ dầu vào (p=1, s=1), ta thuc được kết quả là 1@10x10. Vì là filter 3x3 nên với mỗi điển trên output ta cần thực hiện 9 phép nhân. 

Số lượng phép nhân đã thực hiện là 5x(3x3x10x10) = 4500. (Phép cộng không đáng kể).

Độ phức tạp còn phụ thuộc vào số lượng bộ lọc, như ví dụ là 64 filter => tổng số phép toán là 54x4500 = 288000.

Rút ra công thức chung, với:
  - M, N là số input , output channels.
  - Df là chiều của feature map (M@Df*Df)
  - Dk là chiều của kernel
=>  tổng số lượng phép nhân cần thực hiện là M * N * Df² * Dk²
### Depthwise Separable Convolutions (DSC)
Là kỹ thuật được ứng dụng trong các thiết bị gọn nhej, hạn chế về phần cứng. DSC cơ bản chia làm 2 phần là DeepWise Convolution (DC) và Pointwise Convolution (PC).
![image](https://github.com/user-attachments/assets/10be4269-5e92-484f-8a39-61bc44e75685)


#### DC
So với vịêc tích chập 5@10*10 với filter 5@3*3 (p=1, s=1) để cho ra output 1@10*10 như ở SC, ta chia nhỏ ra, cụ thể ta sẽ nhân với kernel 1@3*3, và nhân 5 lần như thế. Khi đó ta thu được kết quả là 5@10*10.

Số phép nhân được thực hiện là M * Df² * Dk² = 4500.

#### PC
Tiếp theo, chúng ta sử dụng kết quả ở trên đi tích chập với các bộ lọc có kích thước là 1*1. Số lượng bộ lúc bằng số kênh mà ta muốn thu được sau khi thực hiện conv, như ở đây là 64 channel => sử dụng 64 filter.

Như vậy số phép nhân cần tính là M * N * Df² = 5*64*10*10 = 32000.

=> tổng số phép nhân là 32000 + 4500 = 36500, so với SC, nó giảm đến 8 lần.

Kích thước ảnh càng lớn thì giảm càng nhiều.

Vậy tổng kết, số lượng tính toán đã giảm (1/N + 1/(Dk²)) lần so với SC.

So sánh kiến trúc của SC và DSC
![image](https://github.com/user-attachments/assets/d0a45074-2be7-4f6e-a168-409aeb2bef6a)

## KIẾN TRÚC DỰ KIẾN
![image](https://github.com/user-attachments/assets/6ca84902-a827-4f37-94dc-e1c336f7078f)



## TÀI LIỆU THAM KHẢO
[1] Andrew Waterman, Krste Asanovi´c, EECS Department, University of California, Berkeley: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf
[2] RM, AMBA® AXI™  and ACE™  Protocol Specification, 2011 

