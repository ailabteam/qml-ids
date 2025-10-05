# 02_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import joblib # Để lưu các đối tượng scaler/encoder

# --- Cấu hình ---
INPUT_DATA_PATH = "./output/cicids2017_cleaned.parquet"
OUTPUT_DIR = "./output"
SCALER_PATH = os.path.join(OUTPUT_DIR, "min_max_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "sequenced_data.npz")

# Cấu hình cho việc tạo chuỗi
SEQUENCE_LENGTH = 20  # Độ dài của mỗi chuỗi (phiên)
TIME_WINDOW = 120    # Cửa sổ thời gian (giây) để nhóm các flow

# --- Các hàm thực thi ---

def load_data(path):
    """Tải dữ liệu đã được làm sạch."""
    print(f"Đang tải dữ liệu từ {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} không tồn tại. Hãy chạy script 01_eda.py trước.")
    df = pd.read_parquet(path)
    print(f"Tải thành công {len(df)} dòng.")
    return df

def preprocess_labels(df):
    """Làm sạch và mã hóa nhãn. Chuyển thành bài toán nhị phân."""
    print("Bắt đầu tiền xử lý nhãn...")
    
    # 1. Làm sạch ký tự lạ
    df['Label'] = df['Label'].str.replace('ï¿½', '', regex=False)
    
    # 2. Chuyển thành bài toán phân loại nhị phân (Normal vs Attack)
    # Đây là cách tiếp cận phổ biến và hiệu quả nhất cho bộ dữ liệu này
    df['Label'] = np.where(df['Label'] == 'BENIGN', 0, 1)
    
    print("Phân phối nhãn sau khi chuyển đổi (0: BENIGN, 1: ATTACK):")
    print(df['Label'].value_counts())
    
    return df

def scale_features(train_df, test_df):
    """Chuẩn hóa các đặc trưng số."""
    print("Bắt đầu chuẩn hóa đặc trưng...")
    
    # Xác định các cột đặc trưng (tất cả trừ 'Label')
    features = [col for col in train_df.columns if col != 'Label']
    
    scaler = MinMaxScaler()
    
    # Fit scaler CHỈ trên dữ liệu train
    train_df[features] = scaler.fit_transform(train_df[features])
    
    # Transform trên dữ liệu test
    test_df[features] = scaler.transform(test_df[features])
    
    print(f"Đã chuẩn hóa {len(features)} đặc trưng.")
    
    # Lưu scaler để sử dụng sau này (ví dụ khi deploy)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler đã được lưu tại: {SCALER_PATH}")
    
    return train_df, test_df

def create_sequences(df, sequence_length):
    """Nhóm các flow thành các chuỗi (phiên)."""
    print("Bắt đầu tạo chuỗi dữ liệu...")
    
    # Giả định: Nhóm các flow theo 'Source_IP'. Đây là một heuristic phổ biến.
    # Trong một bài toán thực tế, có thể nhóm theo (Source IP, Destination IP, Destination Port).
    # Để đơn giản, chúng ta sẽ nhóm theo các cửa sổ 20 flow liên tiếp.
    
    features = [col for col in df.columns if col != 'Label']
    X = df[features].values
    y = df['Label'].values
    
    sequences_X = []
    sequences_y = []
    
    # Trượt một cửa sổ có độ dài `sequence_length` trên toàn bộ dữ liệu
    for i in tqdm(range(len(X) - sequence_length + 1), desc="Tạo chuỗi"):
        sequences_X.append(X[i:i+sequence_length])
        # Nhãn của chuỗi là nhãn của flow CUỐI CÙNG trong chuỗi
        sequences_y.append(y[i+sequence_length-1])

    return np.array(sequences_X), np.array(sequences_y)


def main():
    """Hàm chính thực thi quy trình tiền xử lý."""
    df = load_data(INPUT_DATA_PATH)
    
    # Tạm thời chỉ lấy một phần dữ liệu để chạy nhanh hơn khi debug
    # BỎ COMMENT DÒNG DƯỚI KHI CHẠY THẬT
    # df = df.sample(n=500000, random_state=42) 
    
    df = preprocess_labels(df)
    
    # Tách X và y
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Chia train/test (80/20)
    # stratify=y để đảm bảo tỉ lệ các lớp được giữ nguyên trong cả 2 tập
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Kích thước tập Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Gắn lại y vào X để scaling và tạo chuỗi
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Chuẩn hóa đặc trưng
    # Lưu ý: Hàm này thay đổi trực tiếp trên DataFrame
    train_df_scaled, test_df_scaled = scale_features(train_df, test_df)
    
    # Tạo chuỗi cho tập train và test
    X_train_seq, y_train_seq = create_sequences(train_df_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(test_df_scaled, SEQUENCE_LENGTH)
    
    print("\n--- Kích thước dữ liệu cuối cùng ---")
    print(f"X_train_seq shape: {X_train_seq.shape}") # (số mẫu, độ dài chuỗi, số đặc trưng)
    print(f"y_train_seq shape: {y_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")
    print(f"y_test_seq shape: {y_test_seq.shape}")

    # Lưu dữ liệu đã xử lý
    np.savez(
        PROCESSED_DATA_PATH,
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_test=X_test_seq,
        y_test=y_test_seq
    )
    print(f"Dữ liệu đã được tiền xử lý và lưu tại: {PROCESSED_DATA_PATH}")


if __name__ == '__main__':
    main()
