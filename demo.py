#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime  # type:ignore

from util import (
    region_dict,
    hiragana_dict,
    class_num_01_dict,
    class_num_02_dict,
    class_num_03_dict,
    run_lpd_inference,
    run_lpr_inference,
    CvDrawText,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument(
        "--lpd",
        type=str,
        # default="weight/PlateYOLO-JP-320x320.onnx",
        default="weight/PlateYOLO-JP-640x640.onnx",
        # default="weight/PlateYOLO-JP-1280x1280.onnx",
        # default="weight/PlateYOLO-JP-1920x1920.onnx",
    )
    parser.add_argument(
        "--lpr",
        type=str,
        default="weight/EkMixer-128x128.onnx",
    )

    parser.add_argument("--lpd_score_th", type=float, default=0.3)
    parser.add_argument("--lpr_min_width1", type=int, default=110)
    parser.add_argument("--lpr_min_width2", type=int, default=150)

    parser.add_argument("--use_video_writer", action="store_true")
    parser.add_argument("--output", type=str, default="output.avi")

    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--use_privacy_mode", action="store_true")

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args = get_args()

    device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    if args.video is not None:
        video_path: str = args.video
    image_path: str = args.image

    lpd_model_path: str = args.lpd
    lpr_model_path: str = args.lpr

    lpd_score_th: float = args.lpd_score_th
    lpr_min_width1: int = args.lpr_min_width1
    lpr_min_width2: int = args.lpr_min_width2

    providers: list[str] = ["CPUExecutionProvider"]
    if args.use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    use_video_writer: bool = args.use_video_writer
    output_path: str = args.output

    use_privacy_mode: bool = args.use_privacy_mode

    # ONNXモデルの読み込み
    lpd_model = onnxruntime.InferenceSession(lpd_model_path, providers=providers)
    lpr_model = onnxruntime.InferenceSession(lpr_model_path, providers=providers)

    # VideoCapture準備
    if args.video is None:
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        cap = cv2.VideoCapture(video_path)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter準備
    video_writer = None

    # ウォームアップ
    _ = run_lpd_inference(lpd_model, np.zeros((960, 540, 3), dtype=np.uint8), 0.3)
    _ = run_lpr_inference(lpr_model, np.zeros((200, 100, 3), dtype=np.uint8))

    while True:
        if image_path is None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(image_path)
        frame_height, frame_width = frame.shape[:2]

        # ナンバープレート検出
        lpd_start_time = time.perf_counter()
        detection_results = run_lpd_inference(lpd_model, frame, lpd_score_th)
        lpd_end_time = time.perf_counter()
        lpd_elapsed_time = (lpd_end_time - lpd_start_time) * 1000

        # ナンバープレート認識
        lpr_start_time = time.perf_counter()
        lpr_results = []
        for detection_result in detection_results:
            # 切り抜き
            offset = 0
            x1: int = int(detection_result[0] * frame_width) - offset
            y1: int = int(detection_result[1] * frame_height) - offset
            x2: int = int(detection_result[2] * frame_width) + offset
            y2: int = int(detection_result[3] * frame_height) + offset
            lp_image = frame[y1:y2, x1:x2]

            if lp_image.shape[0] <= 0 or lp_image.shape[1] <= 0:
                continue

            # 推論
            hiragana_id, region_id, class_num_ids, plate_num_ids = run_lpr_inference(
                lpr_model, lp_image
            )

            lpr_results.append(
                {
                    "bbox": detection_result[:4],
                    "bbox_score": detection_result[4],
                    "bbox_class_id": detection_result[5],
                    "lp_shape": lp_image.shape,
                    "hiragana_id": hiragana_id,
                    "region_id": region_id,
                    "class_num_ids": class_num_ids,
                    "plate_num_ids": plate_num_ids,
                }
            )
        lpr_end_time = time.perf_counter()
        lpr_elapsed_time = (lpr_end_time - lpr_start_time) * 1000

        # デバッグ表示
        debug_image = draw_info(
            frame,
            lpr_results,
            lpd_elapsed_time,
            lpr_elapsed_time,
            region_dict,
            hiragana_dict,
            class_num_01_dict,
            class_num_02_dict,
            class_num_03_dict,
            lpr_min_width1,
            lpr_min_width2,
            use_privacy_mode,
        )

        cv2.imshow("LPR Demo", debug_image)
        if image_path is None:
            if cv2.waitKey(1) == 27:  # ESC
                break
        else:
            cv2.waitKey(-1)
            break

        # 動画書き込み
        if use_video_writer and video_writer is None:
            debug_width = debug_image.shape[1]
            debug_height = debug_image.shape[0]
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                int(cap_fps),
                (debug_width, debug_height),
            )
        if use_video_writer:
            video_writer.write(debug_image)  # type:ignore

    cap.release()
    if use_video_writer and video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def draw_info(
    image,
    lpr_results,
    lpd_elapsed_time,
    lpr_elapsed_time,
    region_dict,
    hiragana_dict,
    class_num_01_dict,
    class_num_02_dict,
    class_num_03_dict,
    lpr_min_width1,
    lpr_min_width2,
    use_privacy_mode,
    resize_width=960,
    font_path="./font/gensen-font/ttc/GenSenRounded2-B.ttc",
):
    debug_image = copy.deepcopy(image)

    # サイズ調整
    debug_image = cv2.resize(
        debug_image,
        (resize_width, int(debug_image.shape[0] * resize_width / debug_image.shape[1])),
    )
    image_height, image_width = debug_image.shape[:2]

    # ナンバープレートID定義
    region_dict_inv = {v: k for k, v in region_dict.items()}
    hiragana_dict_inv = {v: k for k, v in hiragana_dict.items()}
    class_num_01_dict_inv = {v: k for k, v in class_num_01_dict.items()}
    class_num_02_dict_inv = {v: k for k, v in class_num_02_dict.items()}
    class_num_03_dict_inv = {v: k for k, v in class_num_03_dict.items()}

    for lpr_result in lpr_results:
        # バウンディングボックス
        x1 = int(lpr_result["bbox"][0] * image_width)
        y1 = int(lpr_result["bbox"][1] * image_height)
        x2 = int(lpr_result["bbox"][2] * image_width)
        y2 = int(lpr_result["bbox"][3] * image_height)
        if not use_privacy_mode:
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), -1)

        # ナンバープレートサイズ
        lp_shape = lpr_result["lp_shape"]
        lp_width = lp_shape[1]

        # ナンバープレート：地域名
        region_text = region_dict_inv.get(lpr_result["region_id"], 0)

        if lpr_min_width1 > lp_width:
            region_text = ""
        elif lpr_min_width2 > lp_width:
            region_text = "読取不可"

        # ナンバープレート：ひらがな
        hiragana_text = hiragana_dict_inv.get(lpr_result["hiragana_id"], 0)

        if lpr_min_width1 > lp_width:
            hiragana_text = "読取不可"
        elif lpr_min_width2 > lp_width:
            hiragana_text = "読取不可"

        # ナンバープレート：分類番号
        class_text = ""
        if lpr_result["class_num_ids"][0] < len(class_num_01_dict_inv):
            class_text += class_num_01_dict_inv[lpr_result["class_num_ids"][0]]
        if lpr_result["class_num_ids"][1] < len(class_num_02_dict_inv):
            if not use_privacy_mode:
                class_text += class_num_02_dict_inv[lpr_result["class_num_ids"][1]]
            else:
                class_text += "X"
        if lpr_result["class_num_ids"][2] < len(class_num_03_dict_inv):
            if not use_privacy_mode:
                class_text += class_num_03_dict_inv[lpr_result["class_num_ids"][2]]
            else:
                class_text += "X"

        if lpr_min_width1 > lp_width:
            class_text = ""
        elif lpr_min_width2 > lp_width:
            class_text = "読取不可"

        # ナンバープレート：一連指定番号
        plate_text = ""
        for plate_num_index, plate_value in enumerate(lpr_result["plate_num_ids"]):
            if not use_privacy_mode:
                if plate_value != 10:
                    plate_text += str(plate_value)
            else:
                if plate_num_index < 3:
                    plate_text += "X"
                else:
                    if plate_value != 10:
                        plate_text += str(plate_value)

        if lpr_min_width1 > lp_width:
            plate_text = ""

        # ナンバープレート情報描画
        # debug_image = CvDrawText.puttext(
        #     debug_image,
        #     "width:" + str(lp_width),
        #     (x1, y1 - 45 -2),
        #     font_path,
        #     15,
        #     (0, 255, 0),  # RGB
        # )
        debug_image = CvDrawText.puttext(
            debug_image,
            region_text + " " + class_text,
            (x1, y1 - 30 - 2),
            font_path,
            15,
            (0, 255, 0),  # RGB
        )
        debug_image = CvDrawText.puttext(
            debug_image,
            hiragana_text + " " + plate_text,
            (x1, y1 - 15 - 2),
            font_path,
            15,
            (0, 255, 0),  # RGB
        )

    # 処理時間
    lpr_count = len(lpr_results)
    lpr_mean_time = (lpr_elapsed_time / lpr_count) if lpr_count > 0 else 0
    debug_image = CvDrawText.puttext(
        debug_image,
        f"Total:{lpd_elapsed_time + lpr_elapsed_time:.0f}ms",
        (5, 5),
        font_path,
        15,
        (0, 255, 0),  # RGB
    )
    debug_image = CvDrawText.puttext(
        debug_image,
        f"    LPD:{lpd_elapsed_time:.0f}ms",
        (5, 20),
        font_path,
        15,
        (0, 255, 0),  # RGB
    )
    debug_image = CvDrawText.puttext(
        debug_image,
        f"    LPR:{lpr_elapsed_time:.0f}ms(count:{lpr_count}, avg:{lpr_mean_time:.0f}ms)",
        (5, 35),
        font_path,
        15,
        (0, 255, 0),  # RGB
    )

    return debug_image


if __name__ == "__main__":
    main()
