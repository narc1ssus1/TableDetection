# -*- encoding: utf-8 -*-
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any
from numpy import ndarray

from .table_line_rec import TableLineRecognition
from .table_recover import TableRecover
from .utils import InputType, LoadImage
from .utils_table_recover import plot_html_table

cur_dir = Path(__file__).resolve().parent
default_model_path = cur_dir / "models" / "cycle_center_net_v1.onnx"


class WiredTableRecognition:
    def __init__(self, table_model_path: Union[str, Path] = default_model_path):
        self.load_img = LoadImage()
        self.table_line_rec = TableLineRecognition(str(table_model_path))
        self.table_recover = TableRecover()

    def __call__(self, img: InputType) -> tuple[str, float, list[Any]] | tuple[str, float, ndarray | ndarray]:
        s = time.perf_counter()

        img = self.load_img(img)
        polygons = self.table_line_rec(img)
        if polygons is None:
            logging.warning("polygons is None.")
            return "", 0.0,[]

        try:
            table_res = self.table_recover(polygons)
            table_str = plot_html_table(table_res, {})
            elapse = time.perf_counter() - s
        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0, polygons
        else:
            return table_str, elapse,polygons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, required=True)
    args = parser.parse_args()

    table_rec = WiredTableRecognition()
    table_str, elapse = table_rec(args.img_path)
    print(table_str)
    print(f"cost: {elapse:.5f}")


if __name__ == "__main__":
    main()