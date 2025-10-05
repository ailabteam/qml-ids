# 03b_train_qlstm_q8.py
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
# --- THAY ĐỔI CHÍNH Ở ĐÂY ---
N_QUBITS = 8 # TĂNG SỐ QUBITS
Q_DEPTH = 2
DEVICE_NAME = "default.qubit"

# Phần Huấn luyện
DATA_PATH = "./output/sequenced_data.npz"
OUTPUT_DIR = "./output"
# --- THAY ĐỔI TÊN FILE OUTPUT ---
MODEL_PATH = os.path.join(OUTPUT_DIR, "qlstm_model_q8.pth")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "figures/confusion_matrix_q8.png")

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 1. Định nghĩa Mạch Lượng tử (Giữ nguyên logic, chỉ thay đổi số qubits) ---
q_device = qml.device(DEVICE_NAME, wires=N_QUBITS)
@qml.qnode(q_device, interface='torch')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# --- 2. Định nghĩa Mô hình Hybrid QLSTM ---
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, q_depth):
        super(QLSTM, self).__init__()
        self.n_qubits = n_qubits
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.clayer = nn.Linear(hidden_size, n_qubits) # Tự động điều chỉnh output size
        weight_shapes = {"weights": (q_depth, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.output_layer = nn.Linear(1, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        lstm_out = h_n.squeeze(0)
        classical_out = self.clayer(lstm_out)
        quantum_out = self.qlayer(classical_out).unsqueeze(1)
        output = self.output_layer(quantum_out)
        return output

# --- 3. Vòng lặp Huấn luyện và Đánh giá (Giữ nguyên) ---
def train(model, train_loader, optimizer, criterion):
    # ... (code y hệt file 03)
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
    # ... (code y hệt file 03)
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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'])
    # --- THAY ĐỔI TITLE ---
    plt.title('Confusion Matrix (QLSTM, n_qubits=8)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, dpi=600)
    plt.close()

# --- 4. Hàm Main ---
def main():
    print("Đang tải dữ liệu đã tiền xử lý...")
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = X_train.shape[2]
    hidden_size = 32 # Giữ nguyên hidden_size
    # Model sẽ tự động điều chỉnh clayer cho 8 qubits
    model = QLSTM(input_size, hidden_size, N_QUBITS, Q_DEPTH).to(DEVICE)
    print("Thử nghiệm B: Tăng số qubits")
    print("Kiến trúc mô hình:")
    print(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = train(model, train_loader, optimizer, criterion)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {end_time - start_time:.2f}s")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Mô hình đã được lưu tại: {MODEL_PATH}")
    
    y_pred, y_true = evaluate(model, test_loader)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n--- Kết quả Đánh giá (Thử nghiệm B: n_qubits=8) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    print(f"Ma trận nhầm lẫn đã được lưu tại: {CONFUSION_MATRIX_PATH}")

if __name__ == '__main__':
    main()
