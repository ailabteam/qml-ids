# 04_train_lstm_baseline.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm # Đảm bảo đã import tqdm

# --- Cấu hình ---
# Phần Huấn luyện
DATA_PATH = "./output/sequenced_data.npz"
OUTPUT_DIR = "./output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "lstm_baseline_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "figures/confusion_matrix_baseline.png")

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 5 # Giữ nguyên số epochs để so sánh công bằng

# Kiểm tra sự tồn tại của GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 2. Định nghĩa Mô hình LSTM Cổ điển (Baseline) ---
class ClassicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClassicLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Lớp LSTM cổ điển
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Thay thế phần lượng tử bằng một khối MLP cổ điển
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Chạy qua LSTM
        _, (h_n, _) = self.lstm(x)
        lstm_out = h_n.squeeze(0) # Shape: (batch, hidden_size)
        
        # Chạy qua bộ phân loại MLP
        output = self.classifier(lstm_out) # Shape: (batch, 1)
        
        return output

# --- 3. Vòng lặp Huấn luyện và Đánh giá (Giữ nguyên) ---
def train(model, train_loader, optimizer, criterion):
    # (Code giống hệt script 03)
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader):
    # (Code giống hệt script 03)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds).flatten(), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path):
    # (Code giống hệt script 03)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'])
    plt.title('Confusion Matrix (Baseline LSTM)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, dpi=600)
    plt.close()

# --- 4. Hàm Main (Chỉ thay đổi tên model) ---
def main():
    print("Đang tải dữ liệu đã tiền xử lý...")
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Chuyển NumPy array thành PyTorch Tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Khởi tạo mô hình ClassicLSTM
    input_size = X_train.shape[2]
    hidden_size = 32
    model = ClassicLSTM(input_size, hidden_size).to(DEVICE)
    print("Kiến trúc mô hình Baseline:")
    print(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = train(model, train_loader, optimizer, criterion)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {end_time - start_time:.2f}s")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Mô hình Baseline đã được lưu tại: {MODEL_PATH}")
    
    y_pred, y_true = evaluate(model, test_loader)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n--- Kết quả Đánh giá (Baseline) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    print(f"Ma trận nhầm lẫn đã được lưu tại: {CONFUSION_MATRIX_PATH}")

if __name__ == '__main__':
    main()
