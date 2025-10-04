# QML-IDS: Môi trường Docker Bất Tử cho Nghiên cứu Quantum Machine Learning

Dự án này cung cấp một môi trường Docker đã được cấu hình sẵn, ổn định và có thể tái lập 100% cho các nghiên cứu về Quantum Machine Learning (QML), đặc biệt là các mô hình lai (Hybrid Quantum-Classical) sử dụng PyTorch và PennyLane.

Môi trường này được thiết kế để giải quyết "cơn ác mộng cài đặt" thường gặp khi làm việc với các thư viện tính toán hiệu năng cao, đảm bảo rằng bạn có thể bắt đầu nghiên cứu ngay lập tức.

## ✨ Kiến trúc Môi trường

Môi trường này được tối ưu hóa cho các bài toán thực tế, áp dụng kiến trúc hybrid:

*   **PyTorch (trên GPU):** Tận dụng toàn bộ sức mạnh của GPU NVIDIA thông qua CUDA để tăng tốc các phần tính toán cổ điển (chiếm >95% khối lượng công việc), như các lớp mạng nơ-ron sâu.
*   **PennyLane (trên CPU):** Sử dụng trình mô phỏng `default.qubit` an toàn và ổn định. Điều này đảm bảo khả năng tương thích tối đa và tránh các lỗi cấp thấp (`Illegal instruction`), trong khi vẫn đủ nhanh để mô phỏng các mạch lượng tử có kích thước phù hợp cho nghiên cứu (4-16 qubits).

## 🚀 Hướng dẫn sử dụng nhanh (Quick Start)

**Yêu cầu:**
*   Hệ điều hành Linux
*   Docker Engine
*   GPU NVIDIA
*   NVIDIA Driver tương thích
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Các bước thực hiện:**

1.  **Kéo (Pull) Image từ Docker Hub:**
    ```bash
    docker pull haodpsut/qml-ids:1.0
    ```

2.  **Chạy (Run) Container:**
    Di chuyển đến thư mục dự án trên máy của bạn và chạy lệnh sau:
    ```bash
    # Lệnh này sẽ khởi động container, cấp quyền truy cập GPU,
    # và mount thư mục hiện tại của bạn vào /app bên trong container.
    docker run --rm -it --gpus all -v $(pwd):/app haodpsut/qml-ids:1.0
    ```
    Sau khi chạy, bạn sẽ ở bên trong shell của container với môi trường đã được tự động kích hoạt. Dấu nhắc lệnh sẽ có dạng `(/opt/env) root@...:/app#`.

3.  **Xác minh Môi trường:**
    Để chắc chắn mọi thứ hoạt động, hãy chạy script kiểm tra:
    ```bash
    python check_env.py
    ```
    Bạn sẽ thấy thông báo `🎉 SUCCESS!` ở cuối cùng. Môi trường của bạn đã sẵn sàng!

## ❤️ Hành trình Xây dựng Môi trường: Các Bài học Xương máu

Việc tạo ra môi trường này là một quá trình gỡ lỗi đầy thử thách. Phần này ghi lại các vấn đề cốt lõi đã gặp phải và cách chúng được giải quyết, hy vọng sẽ giúp ích cho những người đi sau.

#### Vấn đề 1: Lỗi `Illegal instruction (core dumped)`
*   **Triệu chứng:** Chương trình crash ngay khi gọi đến các hàm của `pennylane-lightning`.
*   **Nguyên nhân gốc:** Các thư viện hiệu năng cao (`pennylane-lightning`, `cuQuantum`) thường được biên dịch sẵn với các tập lệnh CPU hiện đại (AVX, AVX2) để tối ưu hóa tốc độ. Tuy nhiên, nếu CPU của server không hỗ trợ các tập lệnh này, nó sẽ không hiểu và gây ra lỗi. Docker ảo hóa môi trường phần mềm, **nhưng không ảo hóa CPU**, do đó lỗi này vẫn xảy ra.
*   **Bài học:** Phải biên dịch lại thư viện từ mã nguồn trên chính máy đích và ra lệnh tường minh cho trình biên dịch **tắt các tập lệnh AVX** (`-DENABLE_AVX=OFF`).

#### Vấn đề 2: "Dependency Hell" khi Biên dịch từ Source
*   **Triệu chứng:** Quá trình build từ source liên tục thất bại với các lỗi khó hiểu.
*   **Nguyên nhân gốc:** Việc build một thư viện C++/CUDA phức tạp đòi hỏi một chuỗi công cụ hoàn chỉnh:
    1.  **Đúng trình biên dịch C++:** Chúng tôi phát hiện ra `g++ 12` là phiên bản ổn định nhất, không quá cũ cũng không quá mới.
    2.  **Đầy đủ Headers:** Cần có `python-dev` để C++ có thể "nói chuyện" với Python.
    3.  **Toàn bộ CUDA Toolkit:** Cần có `nvcc` để biên dịch mã CUDA, chứ không chỉ `CUDA runtime` để chạy.
    4.  **Các SDK phụ trợ:** `pennylane-lightning-gpu` bắt buộc phải "thấy" `cuQuantum SDK` trong quá trình build, ngay cả khi chúng ta không muốn dùng nó.
*   **Bài học:** Việc cố gắng "chắp vá" một môi trường runtime bằng cách cài thêm các công cụ build là không ổn định. Cách tiếp cận đúng là bắt đầu từ một môi trường `devel` hoàn chỉnh, hoặc kiểm soát chặt chẽ từng dependency như chúng tôi đã làm.

#### Giải pháp cuối cùng: Sự ổn định là trên hết
*   Sau nhiều nỗ lực, chúng tôi phát hiện ra rằng ngay cả khi build thành công, thư viện `cuQuantum` của NVIDIA vẫn chứa mã AVX.
*   **Quyết định cuối cùng:** Chúng tôi đã chọn giải pháp thực dụng và ổn định nhất: sử dụng trình mô phỏng `default.qubit` an toàn của PennyLane (chạy trên CPU) và để PyTorch tận dụng GPU cho phần việc nặng nhất.
*   **Đóng gói bằng `conda-pack`:** Thay vì lặp lại quá trình build đầy rủi ro trong Dockerfile, chúng tôi đã tạo một môi trường Conda hoàn hảo trên host, sau đó dùng `conda-pack` để đóng gói **chính xác trạng thái đã hoạt động** đó vào một file tarball, rồi giải nén nó trong Docker. Đây là phương pháp đảm bảo tính tái lập 100%.

