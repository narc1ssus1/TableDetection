import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .lineless_table_process import DetProcess, get_affine_transform_upper_left
from .utils import LoadImage, OrtInferSession

cur_dir = Path(__file__).resolve().parent
detect_model_path = cur_dir / "models" / "lore_detect.onnx"
process_model_path = cur_dir / "models" / "lore_process.onnx"


class LinelessTableRecognition:
    def __init__(self):
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
        self.inp_h = 768
        self.inp_w = 768
        self.det_session = OrtInferSession(detect_model_path)
        self.process_session = OrtInferSession(process_model_path)
        self.load_img = LoadImage()
        self.det_process = DetProcess()

    def __call__(self, content: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        img = self.load_img(content)
        input_info = self.preprocess(img)
        polygons, slct_logi = self.infer(input_info)
        return polygons, slct_logi

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        height, width = img.shape[:2]
        resized_image = cv2.resize(img, (width, height))

        c = np.array([0, 0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform_upper_left(c, s, [self.inp_w, self.inp_h])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (self.inp_w, self.inp_h), flags=cv2.INTER_LINEAR
        )
        inp_image = ((inp_image / 255.0 - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.inp_h, self.inp_w)
        meta = {
            "c": c,
            "s": s,
            "out_height": self.inp_h // 4,
            "out_width": self.inp_w // 4,
        }
        return {"img": images, "meta": meta}

    def infer(self, input: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        hm, st, wh, ax, cr, reg = self.det_session([input["img"]])
        output = {
            "hm": hm,
            "st": st,
            "wh": wh,
            "ax": ax,
            "cr": cr,
            "reg": reg,
        }
        slct_logi_feat, slct_dets_feat, slct_output_dets = self.det_process(
            output, input["meta"]
        )

        slct_output_dets = slct_output_dets.reshape(-1, 4, 2)

        _, slct_logi = self.process_session(
            [slct_logi_feat, slct_dets_feat.astype(np.int64)]
        )
        return slct_output_dets, slct_logi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, required=True)
    args = parser.parse_args()

    table_rec = LinelessTableRecognition()
    polygons, slct_logi = table_rec(args.img_path)
    print("Polygons:", polygons)
    print("Selection Logic:", slct_logi)


if __name__ == "__main__":
    main()