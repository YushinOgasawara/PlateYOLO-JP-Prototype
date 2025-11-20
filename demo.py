#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指定したディレクトリ内の全画像ファイルからナンバープレートを検出し、
“十分大きく写っていて、かつ複数フレーム連続で同じ値になった”
一連指定番号（4桁）だけを CSV に出力するスクリプト。

- 入力: ディレクトリ (--input_dir)
- 出力: CSV ファイル (--csv_output, デフォルト: serial_numbers.csv)
- モデル: PlateYOLO-JP (LPD) + EkMixer (LPR) の ONNX
"""

import argparse
import csv
import os
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime  # type: ignore

from util import run_lpd_inference, run_lpr_inference


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch LPR: extract stable 4-digit serial numbers from all images in a directory."
    )

    # 入力ディレクトリ
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to process",
    )

    # 出力 CSV
    parser.add_argument(
        "--csv_output",
        type=str,
        default="serial_numbers.csv",
        help="Output CSV file path (default: serial_numbers.csv)",
    )

    # モデルパス
    parser.add_argument(
        "--lpd",
        type=str,
        default="weight/PlateYOLO-JP-1920x1920.onnx",
        help="Path to LPD (PlateYOLO-JP) ONNX model",
    )
    parser.add_argument(
        "--lpr",
        type=str,
        default="weight/EkMixer-128x128.onnx",
        help="Path to LPR (EkMixer) ONNX model",
    )

    # 検出スコア閾値
    parser.add_argument(
        "--lpd_score_th",
        type=float,
        default=0.3,
        help="Score threshold for plate detection",
    )

    # プレート幅フィルタ（小さすぎるプレートは無視）
    parser.add_argument(
        "--min_plate_width",
        type=int,
        default=80,  # ★ 少し緩めに
        help="Minimum plate width (pixels) to trust LPR result",
    )

    # 何フレーム連続で同じ番号が出たら「確定」とみなすか
    parser.add_argument(
        "--confirm_frames",
        type=int,
        default=3,
        help="Number of consecutive frames with the same serial required to confirm it",
    )

    # 同じ番号を再度「別イベント」として出力するまでの最小時間差（秒）
    # （ファイル名がタイムスタンプになっている前提。0 なら時間差による再出力制限なし）
    parser.add_argument(
        "--reemit_gap",
        type=float,
        default=10.0,
        help="Minimum seconds between emitting the same serial again as a new event (0 to disable)",
    )

    # GPU を使うかどうか
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use CUDAExecutionProvider if available",
    )

    args = parser.parse_args()
    return args


def extract_serial_number(plate_num_ids: List[int]) -> str:
    """
    plate_num_ids から一連指定番号（4桁）だけを取り出す。
    モデルでは 10 が「空白」を表している前提で、10 は無視する。
    """
    digits = [str(v) for v in plate_num_ids if v != 10]

    # 一連指定番号は通常4桁なので、最後の4桁だけを使う
    if len(digits) >= 4:
        return "".join(digits[-4:])
    else:
        # 読み取りが不完全な場合はスキップしたいので空文字を返す
        return ""


def run_inference_on_frame(
    frame: np.ndarray,
    lpd_model: onnxruntime.InferenceSession,
    lpr_model: onnxruntime.InferenceSession,
    lpd_score_th: float,
) -> Tuple[List[dict], float, float]:
    """
    1枚の画像に対して LPD + LPR を実行し、LPR結果のリストを返す。
    """
    frame_height, frame_width = frame.shape[:2]

    # ナンバープレート検出
    lpd_start_time = time.perf_counter()
    detection_results = run_lpd_inference(lpd_model, frame, lpd_score_th)
    lpd_end_time = time.perf_counter()
    lpd_elapsed_time = (lpd_end_time - lpd_start_time) * 1000.0

    # ナンバープレート認識
    lpr_start_time = time.perf_counter()
    lpr_results: List[dict] = []

    for detection_result in detection_results:
        # 切り抜き
        offset = 0
        x1: int = int(detection_result[0] * frame_width) - offset
        y1: int = int(detection_result[1] * frame_height) - offset
        x2: int = int(detection_result[2] * frame_width) + offset
        y2: int = int(detection_result[3] * frame_height) + offset

        # 一応クリップ
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        lp_image = frame[y1:y2, x1:x2]

        if lp_image.shape[0] <= 0 or lp_image.shape[1] <= 0:
            continue

        # LPR 推論
        hiragana_id, region_id, class_num_ids, plate_num_ids = run_lpr_inference(
            lpr_model, lp_image
        )

        lpr_results.append(
            {
                "bbox": detection_result[:4],
                "bbox_score": detection_result[4],
                "bbox_class_id": detection_result[5],
                "lp_shape": lp_image.shape,  # (H, W, C)
                "hiragana_id": hiragana_id,
                "region_id": region_id,
                "class_num_ids": class_num_ids,
                "plate_num_ids": plate_num_ids,
            }
        )

    lpr_end_time = time.perf_counter()
    lpr_elapsed_time = (lpr_end_time - lpr_start_time) * 1000.0

    return lpr_results, lpd_elapsed_time, lpr_elapsed_time


def collect_image_files(input_dir: str) -> List[str]:
    """
    ディレクトリ以下の全ての画像ファイルパスを再帰的に集める。
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_paths: List[str] = []

    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_paths.append(os.path.join(root, name))

    image_paths.sort()
    return image_paths


def parse_timestamp_from_filename(path: str) -> Optional[float]:
    """
    ファイル名が "1763592881.616423.jpg" のように
    「UNIXタイム(っぽい数値).拡張子」である前提で float に変換して返す。
    そうでない場合は None。
    """
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(base)
    except ValueError:
        return None


def main() -> None:
    args = get_args()

    input_dir: str = args.input_dir
    csv_output: str = args.csv_output
    lpd_model_path: str = args.lpd
    lpr_model_path: str = args.lpr
    lpd_score_th: float = args.lpd_score_th
    min_plate_width: int = args.min_plate_width
    confirm_frames: int = args.confirm_frames
    reemit_gap: float = args.reemit_gap

    # 画像一覧取得
    image_paths = collect_image_files(input_dir)
    if not image_paths:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} image files in {input_dir}")
    print(
        f"Params: lpd_th={lpd_score_th}, min_plate_width={min_plate_width}, "
        f"confirm_frames={confirm_frames}, reemit_gap={reemit_gap}"
    )

    # ONNX providers
    providers: List[str] = ["CPUExecutionProvider"]
    if args.use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ONNXモデルの読み込み
    print("Loading ONNX models...")
    lpd_model = onnxruntime.InferenceSession(lpd_model_path, providers=providers)
    lpr_model = onnxruntime.InferenceSession(lpr_model_path, providers=providers)

    # ウォームアップ
    print("Warming up models...")
    _ = run_lpd_inference(lpd_model, np.zeros((1920, 1080, 3), dtype=np.uint8), 0.3)
    _ = run_lpr_inference(lpr_model, np.zeros((200, 100, 3), dtype=np.uint8))

    # CSV書き込み用バッファ（確定したイベント）
    rows: List[List[str]] = []

    # ★ 時系列の「安定判定」用の状態
    candidate_serial: Optional[str] = None
    candidate_count: int = 0

    confirmed_last_serial: Optional[str] = None
    confirmed_last_time: Optional[float] = None

    print("Start processing images...")
    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{idx}/{total}] Failed to read image: {img_path}")
            continue

        lpr_results, lpd_time, lpr_time = run_inference_on_frame(
            frame, lpd_model, lpr_model, lpd_score_th
        )

        rel_path = os.path.relpath(img_path, input_dir)
        timestamp = parse_timestamp_from_filename(img_path)

        # このフレームで一番大きなプレートを1つだけ使う
        best_serial: Optional[str] = None
        best_width: int = 0

        for lpr_result in lpr_results:
            h, w, _ = lpr_result["lp_shape"]
            if w < min_plate_width:
                # 遠すぎて小さいプレートは無視
                continue

            serial = extract_serial_number(lpr_result["plate_num_ids"])
            if len(serial) != 4:
                continue

            # より幅の大きいプレートだけを採用
            if w > best_width:
                best_width = w
                best_serial = serial

        # このフレームで信頼できる 4桁番号がなければ、
        # ★ 候補はリセットせず、単に「読めなかった」としてスキップ
        if best_serial is None:
            print(
                f"[{idx}/{total}] {rel_path}: no usable 4-digit serial "
                f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms)"
            )
            continue

        # ★ 安定判定ロジック
        if candidate_serial == best_serial:
            candidate_count += 1
        else:
            candidate_serial = best_serial
            candidate_count = 1

        # 何フレーム連続で出たら「安定」とみなすか
        if candidate_count >= confirm_frames:
            # すでに最後に確定した番号と同じかどうか
            if confirmed_last_serial is None:
                # 初回確定
                rows.append([rel_path, best_serial])
                confirmed_last_serial = best_serial
                confirmed_last_time = timestamp
                print(
                    f"[{idx}/{total}] {rel_path}: CONFIRMED serial={best_serial} "
                    f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
                )
            else:
                if best_serial != confirmed_last_serial:
                    # 異なる番号 → 新しい車として出力
                    rows.append([rel_path, best_serial])
                    confirmed_last_serial = best_serial
                    confirmed_last_time = timestamp
                    print(
                        f"[{idx}/{total}] {rel_path}: CONFIRMED NEW serial={best_serial} "
                        f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
                    )
                else:
                    # 同じ番号 → 時間差を見て、十分離れていれば別イベントとして出力
                    if (
                        reemit_gap > 0
                        and timestamp is not None
                        and confirmed_last_time is not None
                    ):
                        gap = timestamp - confirmed_last_time
                        if gap >= reemit_gap:
                            rows.append([rel_path, best_serial])
                            confirmed_last_time = timestamp
                            print(
                                f"[{idx}/{total}] {rel_path}: RE-EMIT same serial={best_serial} "
                                f"(gap {gap:.1f}s, width={best_width})"
                            )
                        else:
                            print(
                                f"[{idx}/{total}] {rel_path}: same confirmed serial={best_serial} "
                                f"(gap {gap:.1f}s, width={best_width})"
                            )
                    else:
                        # 時間情報が取れない場合や reemit_gap=0 の場合は、
                        # すでに確定済みの同じナンバーとしてログだけ出す
                        print(
                            f"[{idx}/{total}] {rel_path}: same confirmed serial={best_serial} "
                            f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
                        )
        else:
            # まだ安定していない候補
            print(
                f"[{idx}/{total}] {rel_path}: candidate serial={best_serial} "
                f"(count={candidate_count}, LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
            )

    # CSV出力
    os.makedirs(os.path.dirname(csv_output) or ".", exist_ok=True)
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "serial_number"])  # ヘッダ
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} stable serial numbers to: {csv_output}")


if __name__ == "__main__":
    main()
