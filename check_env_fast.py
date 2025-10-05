import sys
import torch
import pennylane as qml

print("--- Quick Environment Sanity Check ---")

# 1. Check Python and Libraries
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"PennyLane version: {qml.__version__}")

# 2. Check GPU access for PyTorch
if not torch.cuda.is_available():
    print("❌ ERROR: PyTorch cannot find CUDA. GPU is not available.")
    sys.exit(1)

device_count = torch.cuda.device_count()
print(f"✅ PyTorch CUDA available: True")
print(f"   - Found {device_count} GPU(s).")
print(f"   - GPU 0: {torch.cuda.get_device_name(0)}")

# 3. Check PennyLane Device and PyTorch Interface
try:
    # Check if the high-performance device exists
    dev = qml.device("lightning.gpu", wires=1)
    print("✅ PennyLane device 'lightning.gpu' loaded successfully.")
except qml.DeviceError:
    print("❌ ERROR: PennyLane device 'lightning.gpu' could not be loaded.")
    print("   - Make sure pennylane-lightning[gpu] is installed correctly.")
    sys.exit(1)

try:
    # Define a minimal QNode with the PyTorch interface
    @qml.qnode(dev, interface='torch')
    def minimal_circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    # Just creating the QNode is a test in itself
    print("✅ PennyLane QNode successfully created with 'torch' interface.")

except Exception as e:
    print(f"❌ ERROR: Failed to create PennyLane QNode with PyTorch interface.")
    print(f"   - Details: {e}")
    sys.exit(1)

print("\n🎉 SUCCESS! Your hybrid QML environment is correctly configured.")
