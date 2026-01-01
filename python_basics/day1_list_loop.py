import os

# 1. Giả lập dữ liệu đầu vào
root_dir = "./BraTS2021_Training_Data"
patient_ids = [f"BraTS2021_{i:05d}" for i in range(1, 11)] # Tạo ID từ 00001 đến 00010

# Giả lập danh sách các ID bị lỗi (cần loại bỏ)
black_list = ["BraTS2021_00005", "BraTS2021_00007"]

# 2. Hàm xử lý nòng cốt
def create_clean_datalist(ids, root, skip_list):
    """
    Input: Danh sách ID, thư mục gốc, danh sách đen.
    Output: List các dictionary sạch để đưa vào Swin UNETR.
    """
    clean_data = []
    
    # Kỹ thuật Nâng cao 1: enumerate
    # Giúp lấy cả số thứ tự (index) lẫn giá trị (pid)
    print(f"{'INDEX':<5} | {'PATIENT ID':<20} | {'STATUS'}")
    print("-" * 40)
    
    for idx, pid in enumerate(ids):
        
        # Kỹ thuật Nâng cao 2: Conditional Logic (Lọc rác)
        if pid in skip_list:
            print(f"{idx:<5} | {pid:<20} | ❌ SKIPPED (Corrupted)")
            continue # Bỏ qua vòng lặp này, đi đến người tiếp theo ngay lập tức
            
        # Logic tạo đường dẫn (như bài trước)
        img_path = os.path.join(root, pid, f"{pid}_flair.nii.gz")
        seg_path = os.path.join(root, pid, f"{pid}_seg.nii.gz")
        
        # Kỹ thuật Nâng cao 3: Dictionary Structure cho MONAI
        # MONAI cần input dạng: [{"image": "path/to/img", "label": "path/to/seg"}, ...]
        data_pair = {
            "id": pid,
            "image": img_path,
            "label": seg_path
        }
        
        clean_data.append(data_pair)
        print(f"{idx:<5} | {pid:<20} | ✅ ADDED")
        
    return clean_data

# --- CHẠY THỬ ---
final_dataset = create_clean_datalist(patient_ids, root_dir, black_list)

print("\n" + "="*30)
print(f"Tổng số dữ liệu gốc: {len(patient_ids)}")
print(f"Tổng số dữ liệu sạch: {len(final_dataset)}")
print("Mẫu dữ liệu đầu tiên đưa vào model:")
print(final_dataset[0])