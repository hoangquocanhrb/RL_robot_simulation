# RL_robot_simulation

Mô phỏng robot tìm đường đến đích bằng phương pháp học sâu tăng cường 

# Requirements

* Python 3.8
* Pytorch 1.10.2
* Opencv 4.2.0

# Installation

git clone https://github.com/hoangquocanh/RL_robot_simulation

cd RL_robot_simulation

# Detail

Dự án sử dụng thuật toán A3C (Asynchronous Advantage Actor Critic) và A2C (Advantage Actor Critic) để huấn luyện robot tìm đường tới vị trí mong muốn trong môi trường tĩnh.
Đầu vào là giá trị đo được từ lidar và trạng thái robot hiện tại, đầu ra là hành động của robot. 
# Usage

Chạy mô hình A2C: python3 A2C.py

CHạy mô hình A3C: python3 A3C.py

# Result

Before trained

<img src="result/before.gif">

After trained

<img src="result/after.gif">


