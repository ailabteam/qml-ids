import torch
import pennylane as qml
from pennylane import numpy as np

print("--- PyTorch and CUDA Check ---")
use_gpu = torch.cuda.is_available()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {use_gpu}")
if use_gpu:
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print("-" * 30)

print("\n--- PennyLane + PyTorch GPU Backend Check ---")
if not use_gpu:
    print("Skipping test, CUDA not available.")
else:
    try:
        # Sử dụng 'default.qubit' với giao diện PyTorch
        # và chỉ định torch_device để ép nó chạy trên GPU
        dev = qml.device("default.qubit", wires=2, torch_device="cuda:0")

        # Xây dựng một mạch lượng tử đơn giản
        @qml.qnode(dev, interface="torch")
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        # Tạo một tensor trên GPU
        x = torch.tensor(0.5, device="cuda:0", requires_grad=True)
        
        print("Running quantum circuit on GPU...")
        result = circuit(x)
        result.backward() # Thử tính gradient để kiểm tra
        
        print("\nSUCCESS! PennyLane is using PyTorch backend on GPU.")
        print(f"Circuit result: {result.item()}")
        print(f"Gradient result: {x.grad.item()}")

    except Exception as e:
        print("\nFAILED! An error occurred during the PennyLane GPU test.")
        print(e)
print("-" * 30)
