from flask import Flask, render_template
import torch
import torchvision
from test import CNN

app = Flask(__name__)

# 加载训练好的模型
cnn = CNN()
cnn.load_state_dict(torch.load('cnn2.pkl', map_location=torch.device('cpu')))
cnn.eval()

@app.route('/')
def home():
    return render_template('login.html')  # 指向 login.html 页面

@app.route('/register')
def register():
    return render_template('register.html')  # 指向 register.html 页面

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
