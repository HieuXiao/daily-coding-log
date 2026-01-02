import torch

print("--- DAY 2: PYTORCH TENSOR BASICS ---")

# 1. Giả lập Input của Swin UNETR
# Cấu trúc chuẩn: (Batch_Size, Channels, Depth, Height, Width)
# BraTS có 4 channels: T1, T1ce, T2, FLAIR
# Kích thước crop tiêu chuẩn: 96x96x96 voxel
input_tensor = torch.randn(1, 4, 96, 96, 96)

print(f"1. Kích thước dữ liệu đầu vào (Input Shape): {input_tensor.shape}")

# 2. Thao tác Slicing (Cắt lớp)
# Giả sử muốn lấy channel FLAIR (vị trí index 3) để xem
flair_image = input_tensor[0, 3, :, :, :] # Lấy bỏ dimension Batch và Channel

print(f"2. Kích thước ảnh FLAIR sau khi tách: {flair_image.shape}")

# 3. Thao tác tính toán (Math Operations)
# Tính giá trị trung bình cường độ điểm ảnh (Intensity Mean)
mean_val = torch.mean(input_tensor)
max_val = torch.max(input_tensor)

print(f"3. Thống kê :: Mean: {mean_val:.4f} | Max: {max_val:.4f}")

# 4. Kiểm tra GPU (Cực quan trọng cho Training sau này)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"4. Thiết bị đang chạy: {device.upper()}")