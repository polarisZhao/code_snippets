# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 
#  app_api.py - ******
#  :: written by  zhaozhichao
#  :: Copyright (c) 2023 zhaozhichao
#  :: MIT License, see LICENSE for more details.
#

import base64
import io

from flask import Flask, request, jsonify

app = Flask(__name__)


# 解析 json 数据
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# 用处: 用于进行前端界面和后端传递 json 字符串 
# methods: POST
# Body: raw  
# 类型选择 JSON  
# 示例:
# {
#    "schema": "json",
#    "history": [{"role":"user", "content":"山东境内最高的山是什么山?"}, {"role":"assitant", "content":"泰山"}],
#    "query": "它的海拔高度是多少?"
# }
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
@app.route('/json', methods=['POST'])
def parse_json():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    return jsonify({'received': data}), 200


# 解析表单数据
@app.route('/form', methods=['POST'])
def parse_form():
    data = request.form
    return jsonify({'received': data.to_dict()}), 200

# 解析 URL 查询参数
@app.route('/query', methods=['GET'])
def parse_query():
    param = request.args.get('param_name')
    return jsonify({'received': param}), 200

# 解析上传文件
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# 用处: 用于进行前端界面和后端进行文件传输 
# methods: POST
# Body: form-data
# key: "file"   
# 类型选择 File  
# Value:上传文件即可
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    # 处理文件，例如保存
    return jsonify({'filename': file.filename}), 200



# 解析原始请求体数据
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# 用处: 用于进行前端界面和后端传递字符串 
# methods: POST
# Body: raw  
# 类型选择 Text 
# 示例:
# 泰山的海拔高度是多少?
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
@app.route('/raw', methods=['POST'])
def parse_raw():
    raw_data = request.get_data()
    decoded_data = raw_data.decode('utf-8')
    return jsonify({'received': decoded_data}), 200

# @app.route("/upload_image", methods=['POST'])
# def get_image():
#     img_data = base64.b64decode(str(request.form['image']))
#     # image = Image.open(io.BytesIO(img_data))
#     # print("...get....")

#     # cv::imwrite("test.jpg", image)
#     return "Hello"



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8001)
