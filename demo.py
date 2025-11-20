#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指定したディレクトリ内の全画像ファイルからナンバープレートを検出し、
“十分大きく写っていて、かつ複数フレーム連続で同じ値になった”
一連指定番号（最大4桁）だけを CSV に出力するスクリプト。

確定したナンバーについては、元画像のナンバープレート付近に
そのナンバーをテキスト描画して保存し、そのパスも CSV に書き込む。

さらに、
- ナンバープレートが写っているフレームが連続している区間（segment）ごとに見て、
- その segment 内で一度も CONFIRM にならなかった場合、
  その segment の中で一番プレートが大きかったフレームを "UNREAD"（赤文字）として保存する。

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

# ========= ターミナル用カラー定義 =========
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
# ======================================


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch LPR: extract stable serial numbers (1–4 digits) from all images in a directory."
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

    # 注釈付き画像の保存先ディレクトリ
    parser.add_argument(
        "--annotated_dir",
        type=str,
        default="annotated_plates",
        help="Directory to save annotated images",
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
        default=40,
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
    plate_num_ids から一連指定番号（最大4桁）だけを取り出す。
    モデルでは 10 が「空白」を表している前提で、10 は無視する。
    1桁以上読めていれば、その末尾4桁までを返す。
    """
    digits = [str(v) for v in plate_num_ids if v != 10]

    if len(digits) >= 1:
        # 念のため末尾4桁に制限（5桁以上が来た場合の保険）
        return "".join(digits[-4:])
    else:
        # 読み取りが完全に失敗した場合だけスキップしたいので空文字を返す
        return ""


def run_inference_on_frame(
    frame: np.ndarray,
    lpd_model: onnxruntime.InferenceSession,
    lpr_model: onnxruntime.InferenceSession,
    lpd_score_th: float,
) -> Tuple[List[dict], float, float]:
    """
    1枚の画像に対して LPD + LPR を実行し、LPR結果のリストを返す。
    各結果には、ピクセル座標での bbox も含める。
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
                "bbox": detection_result[:4],            # 正規化座標
                "bbox_px": (x1, y1, x2, y2),             # ピクセル座標
                "bbox_score": detection_result[4],
                "bbox_class_id": detection_result[5],
                "lp_shape": lp_image.shape,              # (H, W, C)
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


def save_annotated_image(
    frame: np.ndarray,
    bbox_px: Tuple[int, int, int, int],
    annotated_dir: str,
    rel_path: str,
    serial: str,
) -> str:
    """
    フレームのナンバープレート付近に serial を文字で描画し、
    画像を保存して、そのパス（相対パス）を返す。

    serial に "UNREAD" が来た場合は赤文字、それ以外（1〜4桁の番号）は緑文字で描画する。
    """
    os.makedirs(annotated_dir, exist_ok=True)

    x1, y1, x2, y2 = bbox_px
    annotated = frame.copy()

    # テキストを書く位置（ナンバープレートの上あたり）
    text = serial
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # テキストサイズを計算して、ちょっといい感じの位置に置く
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = max(0, x1)
    text_y = max(text_h + 5, y1 - 5)

    # UNREAD だけ赤にする
    if serial == "UNREAD":
        text_color = (0, 0, 255)  # 赤 (B, G, R)
    else:
        text_color = (0, 255, 0)  # 緑

    # 文字が見やすいように、薄い黒枠の上に色文字
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    base_name = os.path.splitext(os.path.basename(rel_path))[0]
    out_name = f"{base_name}_annotated_{serial}.jpg"
    out_path = os.path.join(annotated_dir, out_name)

    cv2.imwrite(out_path, annotated)

    # CSV にはプロジェクトルートからの相対パスで入れておく
    return os.path.relpath(out_path)


def main() -> None:
    args = get_args()

    input_dir: str = args.input_dir
    csv_output: str = args.csv_output
    annotated_dir: str = args.annotated_dir
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

    # ★ segment（プレートが写っている塊）管理用
    segment_active: bool = False
    segment_has_confirmed: bool = False
    segment_best_unread_img_path: Optional[str] = None
    segment_best_unread_rel_path: Optional[str] = None
    segment_best_unread_bbox: Optional[Tuple[int, int, int, int]] = None
    segment_best_unread_width: int = 0

    def flush_unread_segment() -> None:
        """現在の segment に対して、UNREAD を1枚保存（必要なら）。"""
        nonlocal segment_active, segment_has_confirmed
        nonlocal segment_best_unread_img_path, segment_best_unread_rel_path
        nonlocal segment_best_unread_bbox, segment_best_unread_width

        if (not segment_active) or segment_has_confirmed:
            return
        if (
            segment_best_unread_img_path is None
            or segment_best_unread_rel_path is None
            or segment_best_unread_bbox is None
        ):
            return

        frame_unread = cv2.imread(segment_best_unread_img_path)
        if frame_unread is None:
            print(
                f"  (WARN) failed to read frame for UNREAD: {segment_best_unread_img_path}"
            )
            return

        unread_rel_path = save_annotated_image(
            frame_unread,
            segment_best_unread_bbox,
            annotated_dir,
            segment_best_unread_rel_path,
            "UNREAD",
        )
        print(
            COLOR_RED
            + f"  -> UNREAD saved for segment: {segment_best_unread_rel_path} "
            f"(width={segment_best_unread_width}, path={unread_rel_path})"
            + COLOR_RESET
        )

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

        has_plate = len(lpr_results) > 0

        # ===== segment への出入りを管理 =====
        if not has_plate:
            # プレートが1枚も検出されなかった → segment 終了の可能性
            if segment_active:
                # 直前までプレートが写っていた塊がここで終わる
                flush_unread_segment()
                # segment リセット
                segment_active = False
                segment_has_confirmed = False
                segment_best_unread_img_path = None
                segment_best_unread_rel_path = None
                segment_best_unread_bbox = None
                segment_best_unread_width = 0
                candidate_serial = None
                candidate_count = 0

            print(
                f"[{idx}/{total}] {rel_path}: no plate detected "
                f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms)"
            )
            continue
        else:
            # プレートが少なくとも1枚検出されている
            if not segment_active:
                # 新しい segment の開始
                segment_active = True
                segment_has_confirmed = False
                segment_best_unread_img_path = None
                segment_best_unread_rel_path = None
                segment_best_unread_bbox = None
                segment_best_unread_width = 0
                candidate_serial = None
                candidate_count = 0

            # このフレームで最大幅のプレート（UNREAD 用候補）
            frame_max_plate = max(lpr_results, key=lambda r: r["lp_shape"][1])
            frame_max_width = frame_max_plate["lp_shape"][1]
            frame_max_bbox = frame_max_plate["bbox_px"]

            # まだこの segment で CONFIRM が出ていない場合のみ、UNREAD 候補を更新
            if not segment_has_confirmed:
                if frame_max_width > segment_best_unread_width:
                    segment_best_unread_width = frame_max_width
                    segment_best_unread_img_path = img_path
                    segment_best_unread_rel_path = rel_path
                    segment_best_unread_bbox = frame_max_bbox

        # ===== ここからは「プレートはある」状態 =====

        # このフレームで一番大きな「使えるプレート」を探す
        best_serial: Optional[str] = None
        best_width: int = 0
        best_bbox_px: Optional[Tuple[int, int, int, int]] = None

        for lpr_result in lpr_results:
            h, w, _ = lpr_result["lp_shape"]
            if w < min_plate_width:
                # 遠すぎて小さいプレートは無視
                continue

            serial = extract_serial_number(lpr_result["plate_num_ids"])
            if len(serial) == 0:
                # 1桁も読めていない → 無視
                continue

            # より幅の大きいプレートだけを採用
            if w > best_width:
                best_width = w
                best_serial = serial
                best_bbox_px = lpr_result["bbox_px"]

        # このフレームで信頼できる番号がなければ
        if best_serial is None or best_bbox_px is None:
            print(
                f"[{idx}/{total}] {rel_path}: plate detected but no usable serial "
                f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, max_width={frame_max_width})"
            )
            # segment_active のまま → 後で flush_unread_segment される可能性あり
            continue

        # ===== 安定判定ロジック =====
        if candidate_serial == best_serial:
            candidate_count += 1
        else:
            candidate_serial = best_serial
            candidate_count = 1

        # 何フレーム連続で出たら「安定」とみなすか
        if candidate_count >= confirm_frames:
            segment_has_confirmed = True  # この segment では少なくとも1回 CONFIRM 出た

            if confirmed_last_serial is None:
                # 初回確定 → CSV に書く & 注釈付き画像を保存
                annotated_rel_path = save_annotated_image(
                    frame, best_bbox_px, annotated_dir, rel_path, best_serial
                )
                rows.append([rel_path, best_serial, annotated_rel_path])
                confirmed_last_serial = best_serial
                confirmed_last_time = timestamp
                print(
                    COLOR_GREEN
                    + f"[{idx}/{total}] {rel_path}: CONFIRMED serial={best_serial} "
                    f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
                    + COLOR_RESET
                )
            else:
                if best_serial != confirmed_last_serial:
                    # 異なる番号 → 新しい車として出力
                    annotated_rel_path = save_annotated_image(
                        frame, best_bbox_px, annotated_dir, rel_path, best_serial
                    )
                    rows.append([rel_path, best_serial, annotated_rel_path])
                    confirmed_last_serial = best_serial
                    confirmed_last_time = timestamp
                    print(
                        COLOR_GREEN
                        + f"[{idx}/{total}] {rel_path}: CONFIRMED NEW serial={best_serial} "
                        f"(LPD:{lpd_time:.0f}ms, LPR:{lpr_time:.0f}ms, width={best_width})"
                        + COLOR_RESET
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
                            annotated_rel_path = save_annotated_image(
                                frame, best_bbox_px, annotated_dir, rel_path, best_serial
                            )
                            rows.append([rel_path, best_serial, annotated_rel_path])
                            confirmed_last_time = timestamp
                            print(
                                COLOR_YELLOW
                                + f"[{idx}/{total}] {rel_path}: RE-EMIT same serial={best_serial} "
                                f"(gap {gap:.1f}s, width={best_width})"
                                + COLOR_RESET
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

    # ===== ループ終了時：最後の segment の後処理 =====
    if segment_active and not segment_has_confirmed:
        flush_unread_segment()

    # CSV出力
    os.makedirs(os.path.dirname(csv_output) or ".", exist_ok=True)
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "serial_number", "annotated_image_path"])  # ヘッダ
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} stable serial numbers to: {csv_output}")


if __name__ == "__main__":
    main()
