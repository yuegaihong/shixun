# 基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 将项目文件复制到容器中
COPY . /app

# 安装所需库
RUN pip install torch torchvision flask opencv-python-headless

# 暴露 Flask 默认端口
EXPOSE 5000

# 启动 Flask 应用
CMD ["python", "app.py"]
