# -*- encoding: utf-8 -*-
from pathlib import Path

from lineless_table_rec import LinelessTableRecognition

engine = LinelessTableRecognition()

img_path = "tests/test_files/lineless_table_recognition.jpg"
table_str, elapse = engine(img_path)
print("table_str:", table_str)
print("elapse:", elapse)

if isinstance(table_str, str):
    print(table_str)
    with open(f"{Path(img_path).stem}.html", "w", encoding="utf-8") as f:
        f.write(table_str)
else:
    print("table_str is not a string. It is of type:", type(table_str))

print("ok")
