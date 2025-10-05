# 03_train_qlstm.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm

# --- Cấu hình ---
# Phần Lượng tử
N_QUBITS = 4          # Số qubits trong mạch lượng tử. Con số nhỏ để chạy nhanh.
Q_DEPTH = 2           # Số lớp (độ sâu) của mạch lượng tử.
DEVICE_NAME = "default.qubit" # Trình mô phỏng an toàn, chạy trên CPU.

# Phần Huấn luyện
DATA_PATH = "./output/sequenced_data.npz"
OUTPUT_DIR = "./output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "qlstm_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "figures/confusion_matrix.png")

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 5 # Bắt đầu với số epochs nhỏ để test, có thể tăng lên sau.

# Kiểm tra sự tồn tại của GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 1. Định nghĩa Mạch Lượng tử (Quantum Circuit) ---
# Đây là trái tim của phần lượng tử.
q_device = qml.device(DEVICE_NAME, wires=N_QUBITS)

@qml.qnode(q_device, interface='torch')
def quantum_circuit(inputs, weights):
    """
    Mạch lượng tử tham số hóa (PQC).
    - inputs: Dữ liệu cổ điển (từ LSTM) được mã hóa vào trạng thái lượng tử.
    - weights: Các tham số có thể học được của mạch (góc quay).
    """
    # Mã hóa dữ liệu: Dùng inputs để quay các qubit quanh trục Y.
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    
    # Lớp biến đổi lượng tử (ansatz) có thể học
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    
    # Phép đo: Lấy giá trị kỳ vọng của toán tử Pauli-Z trên qubit đầu tiên.
    # Kết quả là một số thực duy nhất.
    return qml.expval(qml.PauliZ(0))

# --- 2. Định nghĩa Mô hình Hybrid QLSTM ---
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, q_depth):
        super(QLSTM, self).__init__()
        
        self.n_qubits = n_qubits
        self.hidden_size = hidden_size
        
        # Lớp LSTM cổ điển: input_size là số đặc trưng (78)
        # hidden_size là kích thước vector đầu ra của LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Lớp cổ điển để giảm chiều dữ liệu từ LSTM xuống vừa bằng số qubits
        self.clayer = nn.Linear(hidden_size, n_qubits)
        
        # Tạo lớp lượng tử từ mạch đã định nghĩa
        weight_shapes = {"weights": (q_depth, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Lớp output cuối cùng
        self.output_layer = nn.Linear(1, 1) # Input là 1 (từ qlayer), output là 1 (logit)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Chạy qua LSTM
        # Lấy output cuối cùng của chuỗi (h_n)
        _, (h_n, _) = self.lstm(x)
        lstm_out = h_n.squeeze(0) # Shape: (batch, hidden_size)
        
        # Chạy qua lớp linear để chuẩn bị cho mạch lượng tử
        classical_out = self.clayer(lstm_out) # Shape: (batch, n_qubits)
        
        # Chạy qua lớp lượng tử
        # PyTorch và PennyLane tự xử lý việc chạy vòng lặp qua batch
        quantum_out = self.qlayer(classical_out).unsqueeze(1) # Shape: (batch, 1)
        
        # Chạy qua lớp output cuối cùng
        output = self.output_layer(quantum_out) # Shape: (batch, 1)
        
        return output

# --- 3. Vòng lặp Huấn luyện và Đánh giá ---
def train(model, train_loader, optimizer, criterion):
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
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            
            # Chuyển output thành dự đoán 0 hoặc 1
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    return np.array(all_preds).flatten(), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, dpi=600)
    plt.close()

# --- 4. Hàm Main ---
def main():
    # Tải dữ liệu
    print("Đang tải dữ liệu đã tiền xử lý...")
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # Chỉ lấy một phần nhỏ dữ liệu để chạy thử nhanh
    # BỎ COMMENT CÁC DÒNG NÀY KHI CHẠY THẬT
    # sample_size_train = 100000 
    # sample_size_test = 20000
    # X_train, y_train = X_train[:sample_size_train], y_train[:sample_size_train]
    # X_test, y_test = X_test[:sample_size_test], y_test[:sample_size_test]
    
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
    
    # Khởi tạo mô hình
    input_size = X_train.shape[2] # Số đặc trưng (78)
    hidden_size = 32 # Kích thước trạng thái ẩn của LSTM
    model = QLSTM(input_size, hidden_size, N_QUBITS, Q_DEPTH).to(DEVICE)
    print("Kiến trúc mô hình:")
    print(model)
    
    # Định nghĩa hàm loss và optimizer
    # BCEWithLogitsLoss ổn định hơn cho bài toán nhị phân
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Huấn luyện
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = train(model, train_loader, optimizer, criterion)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {end_time - start_time:.2f}s")
        
    # Lưu mô hình
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Mô hình đã được lưu tại: {MODEL_PATH}")
    
    # Đánh giá
    y_pred, y_true = evaluate(model, test_loader)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n--- Kết quả Đánh giá ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Vẽ và lưu ma trận nhầm lẫn
    plot_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    print(f"Ma trận nhầm lẫn đã được lưu tại: {CONFUSION_MATRIX_PATH}")

if __name__ == '__main__':
    main()
