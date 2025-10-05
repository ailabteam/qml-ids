# 01_eda.py (Phiên bản gốc - Gộp nhiều file)
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
DATA_DIR = "./data" # Thư mục chứa 8 file CSV
OUTPUT_DIR = "./output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "cicids2017_cleaned.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_and_combine_data(data_dir):
    """Đọc tất cả các file CSV trong thư mục và gộp chúng lại."""
    # Tìm tất cả file có đuôi .csv trong thư mục data
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong thư mục: {data_dir}")
        
    print(f"Tìm thấy {len(csv_files)} file CSV. Bắt đầu gộp...")
    
    df_list = []
    for file in tqdm(csv_files, desc="Đang đọc các file CSV"):
        df_list.append(pd.read_csv(file, encoding='latin1')) # Thêm encoding='latin1' để tránh lỗi
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Gộp thành công. Tổng số dòng: {len(full_df)}")
    return full_df

def clean_dataframe(df):
    """Làm sạch tên cột, xử lý giá trị NaN và Infinity."""
    print("Bắt đầu làm sạch DataFrame...")
    
    # 1. Làm sạch tên cột
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    
    # 2. Xử lý giá trị Infinity và NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Số giá trị NaN/Infinity tìm thấy: {df.isna().sum().sum()}")
    df.dropna(inplace=True)
    print(f"Số dòng còn lại sau khi xóa NaN: {len(df)}")
    
    # 3. Chuyển các cột object (trừ Label) thành số
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Xóa lại NaN có thể phát sinh sau khi chuyển đổi
    df.dropna(inplace=True)
    print(f"Số dòng còn lại sau khi ép kiểu: {len(df)}")

    print("Làm sạch hoàn tất.")
    return df

def analyze_and_plot_labels(df, save_path):
    """Phân tích và vẽ biểu đồ phân phối nhãn."""
    print("Phân tích phân phối nhãn...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    label_counts = df['Label'].value_counts()
    print("Phân phối nhãn:")
    print(label_counts)
    
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Phân phối các lớp trong bộ dữ liệu CIC-IDS2017', fontsize=16)
    plt.ylabel('Số lượng mẫu', fontsize=12)
    plt.xlabel('Lớp', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Biểu đồ phân phối lớp đã được lưu tại: {save_path}")
    plt.close()

def main():
    """Hàm chính thực thi toàn bộ quy trình."""
    df = load_and_combine_data(DATA_DIR)
    cleaned_df = clean_dataframe(df)
    plot_path = os.path.join(FIGURE_DIR, "label_distribution.png")
    analyze_and_plot_labels(cleaned_df, plot_path)
    cleaned_df.to_parquet(PROCESSED_DATA_PATH)
    print(f"DataFrame đã làm sạch được lưu tại: {PROCESSED_DATA_PATH}")
    print("\nQuy trình phân tích dữ liệu ban đầu hoàn tất!")

if __name__ == '__main__':
    main()
