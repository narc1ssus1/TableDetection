# -*- encoding: utf-8 -*-
import os
import tempfile
from wired_table_rec import WiredTableRecognition
import cv2
from flask_cors import CORS
from flask import Flask, request, send_file
import time
# table_rec = WiredTableRecognition()
# img_path = "D:\\招标参数（一次）-13824092815_12.jpg"
# img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
# table_str, elapse, bbox = table_rec(img_path)
# print(table_str)
# print(elapse)
# print(bbox)
# for bbox in bbox:
#     for line in bbox:
#         x1, y1 = int(line[0]), int(line[1])
#         x2 = int(bbox[1][0])
#         y2 = int(bbox[3][1])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv2.imwrite('D://rest.bmp', img)

app = Flask(__name__)
app.json.sort_keys = False
CORS(app, supports_credentials=True)




# 在全局范围内创建并配置 WiredTableRecognition 实例以使用 GPU
table_rec = WiredTableRecognition()


@app.route('/pdf-to-image', methods=['POST'])
def pdf_to_image():
    uploaded_file = request.files['img']

    # 保存文件到临时位置
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        uploaded_file.save(tmp_file)
        tmp_file_path = tmp_file.name

    # 使用 OpenCV 读取图像
    img = cv2.imread(tmp_file_path)

    start_time = time.time()
    table_str, elapse, bbox = table_rec(img)
    print(elapse)

    # 处理并输出边界框
    for bbox in bbox:
        for line in bbox:
            x1, y1 = int(line[0]), int(line[1])
            x2, y2 = int(bbox[1][0]), int(bbox[3][1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 保存带边界框的图像到临时文件
    output_file_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(output_file_path, img)

    # 发送图像文件
    return send_file(output_file_path, mimetype='image/png')


if __name__ == "__main__":
    app.run("0.0.0.0", 5200)