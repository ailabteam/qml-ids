import torch
import pennylane as qml

print("--- Hybrid GPU(PyTorch)/CPU(PennyLane) Environment Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"PennyLane version: {qml.__version__}")

# 1. Kiểm tra GPU cho PyTorch
if not torch.cuda.is_available():
    raise RuntimeError("ERROR: PyTorch cannot find GPU!")
print(f"✅ PyTorch CUDA available: True ({torch.cuda.get_device_name(0)})")

# 2. Định nghĩa thiết bị lượng tử an toàn (chạy trên CPU)
n_qubits = 4
# default.qubit là trình mô phỏng an toàn, không cần build
device = qml.device("default.qubit", wires=n_qubits)
print(f"✅ PennyLane device '{device.name}' loaded successfully (runs on CPU).")

# 3. Định nghĩa QNode và lớp PyTorch
@qml.qnode(device, interface='torch')
def quantum_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Lớp lượng tử có 4 input (cho 4 qubit) và 1 output
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})

# 4. Kiểm tra Forward/Backward pass trên GPU
print("\n--- Testing Hybrid Forward/Backward Pass ---")

# Tạo một lớp cổ điển và lớp lượng tử
# Lớp linear sẽ chạy trên GPU
classical_layer = torch.nn.Linear(10, n_qubits) 
model = torch.nn.Sequential(classical_layer, qlayer)

# Chuyển mô hình lên GPU
gpu_device = torch.device("cuda:0")
model.to(gpu_device)
print(f"✅ Model moved to {gpu_device}")

# Dữ liệu đầu vào trên GPU
inputs = torch.randn(5, 10, device=gpu_device)

# Forward pass
print("Performing forward pass...")
# PyTorch tự động xử lý:
# - inputs -> classical_layer (trên GPU)
# - output của classical_layer -> CPU -> qlayer (trên CPU)
# - output của qlayer -> GPU
result = model(inputs)
print(f"Input shape: {inputs.shape}, Device: {inputs.device}")
print(f"Output shape: {result.shape}, Device: {result.device}")

# Backward pass
result.sum().backward()
print("✅ Backward pass successful.")
print("\n🎉 SUCCESS! Your hybrid GPU/CPU environment is ready for research.")
