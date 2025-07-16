#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import os
import re
from datetime import datetime, date
from typing import Union, List, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # .envファイルを読み込み
except ImportError:
    print("警告: python-dotenvがインストールされていません。.envファイルは無視されます。")
    print("インストール: uv add python-dotenv")

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

# 型エイリアス定義
VideoInput = Union[int, str, List[str]]
VideoCapture = cv2.VideoCapture
InferenceSession = onnxruntime.InferenceSession


def resolve_date_path(path_pattern: str) -> str:
    """
    日付パターンを含むパスを解決する
    
    Args:
        path_pattern: パスパターン（yyyy-mm-dd形式を含む可能性）
    
    Returns:
        解決されたパス
    
    Examples:
        "./videos/2024-01-15/" -> "./videos/2024-01-15/"
        "./videos/{date}/" -> "./videos/2024-07-12/" (今日の日付)
        "./videos/{today}/" -> "./videos/2024-07-12/" (今日の日付)
    """
    # yyyy-mm-dd形式の日付パターンを検出
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    
    # 既に具体的な日付が指定されている場合はそのまま返す
    if re.search(date_pattern, path_pattern):
        return path_pattern
    
    # {date} や {today} パターンを今日の日付に置換
    today_str = date.today().strftime('%Y-%m-%d')
    path_pattern = path_pattern.replace('{date}', today_str)
    path_pattern = path_pattern.replace('{today}', today_str)
    
    return path_pattern


def resolve_video_directory(base_dir: str, date_str: Optional[str] = None) -> str:
    """
    動画ディレクトリパスを解決する
    
    Args:
        base_dir: ベースディレクトリ
        date_str: 日付文字列（yyyy-mm-dd形式、Noneの場合は今日）
    
    Returns:
        解決されたディレクトリパス
    """
    if date_str is None:
        # 環境変数から日付を取得、なければ今日の日付
        date_str = os.getenv('VIDEO_DATE', date.today().strftime('%Y-%m-%d'))
    
    # 日付パターンを検証
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(date_pattern, date_str):
        raise ValueError(f"日付は yyyy-mm-dd 形式で指定してください: {date_str}")
    
    # パスを構築
    resolved_path = Path(base_dir) / date_str
    return str(resolved_path)


def get_env_value(key: str, default: str, convert_type=str):
    """環境変数を型変換して取得"""
    value = os.getenv(key, default)
    if convert_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif convert_type == float:
        return float(value)
    elif convert_type == int:
        return int(value)
    return value


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # 入力ソース（排他的グループ）
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--device", type=int, help="カメラデバイス番号")
    input_group.add_argument("--video", type=str, help="単一動画ファイル")
    input_group.add_argument("--videos", nargs="+", type=str, help="複数動画ファイル")
    input_group.add_argument("--video-dir", type=str, help="動画ディレクトリ")
    input_group.add_argument("--image", type=str, help="画像ファイル")
    
    # 日付指定オプション
    parser.add_argument("--date", type=str, help="動画ディレクトリの日付 (yyyy-mm-dd形式)")
    
    # ヘッドレスモードオプション
    parser.add_argument("--headless", action="store_true", help="ヘッドレスモード（GUI無効）")

    # 表示設定
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    # モデル設定（環境変数対応）
    parser.add_argument(
        "--lpd",
        type=str,
        default=get_env_value("LPD_MODEL_PATH", "weight/PlateYOLO-JP-640x640.onnx"),
        help="LPDモデルパス（環境変数: LPD_MODEL_PATH）"
    )
    parser.add_argument(
        "--lpr",
        type=str,
        default=get_env_value("LPR_MODEL_PATH", "weight/EkMixer-128x128.onnx"),
        help="LPRモデルパス（環境変数: LPR_MODEL_PATH）"
    )

    # 推論設定（環境変数対応）
    parser.add_argument(
        "--lpd_score_th", 
        type=float, 
        default=get_env_value("LPD_SCORE_THRESHOLD", "0.3", float),
        help="LPD信頼度閾値（環境変数: LPD_SCORE_THRESHOLD）"
    )
    parser.add_argument(
        "--lpr_min_width1", 
        type=int, 
        default=get_env_value("LPR_MIN_WIDTH1", "110", int),
        help="LPR最小幅1（環境変数: LPR_MIN_WIDTH1）"
    )
    parser.add_argument(
        "--lpr_min_width2", 
        type=int, 
        default=get_env_value("LPR_MIN_WIDTH2", "150", int),
        help="LPR最小幅2（環境変数: LPR_MIN_WIDTH2）"
    )

    # 出力設定（環境変数対応）
    parser.add_argument("--use_video_writer", action="store_true")
    parser.add_argument("--output", type=str, default="output.avi")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=get_env_value("OUTPUT_BASE_DIR", "./outputs/"), 
        help="複数動画処理時の出力ディレクトリ（環境変数: OUTPUT_BASE_DIR）"
    )

    # その他（環境変数対応）
    parser.add_argument(
        "--use_gpu", 
        action="store_true",
        default=get_env_value("USE_GPU", "false", bool),
        help="GPU使用（環境変数: USE_GPU）"
    )
    parser.add_argument(
        "--use_privacy_mode", 
        action="store_true",
        default=get_env_value("USE_PRIVACY_MODE", "false", bool),
        help="プライバシーモード（環境変数: USE_PRIVACY_MODE）"
    )

    return parser.parse_args()


def determine_input_type(args: argparse.Namespace) -> VideoInput:
    """引数から入力タイプを決定する"""
    if args.device is not None:
        return args.device
    elif args.video is not None:
        return args.video
    elif args.videos is not None:
        return args.videos
    elif args.video_dir is not None:
        # 指定されたディレクトリを使用
        video_dir = Path(resolve_date_path(args.video_dir))
        if not video_dir.exists():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {video_dir}")
        
        # 動画ファイル拡張子
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        video_files = [
            str(f) for f in video_dir.iterdir() 
            if f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            raise ValueError(f"動画ファイルが見つかりません: {video_dir}")
        
        return sorted(video_files)
    elif args.image is not None:
        return args.image
    else:
        # デフォルト: 環境変数または日付指定のディレクトリを使用
        base_dir = get_env_value("VIDEO_BASE_DIR", "./videos")
        
        try:
            if args.date:
                # --date引数で指定された日付を使用
                video_dir = Path(resolve_video_directory(base_dir, args.date))
            else:
                # 環境変数VIDEO_DATEまたは今日の日付を使用
                date_str = os.getenv('VIDEO_DATE')
                if date_str:
                    video_dir = Path(resolve_video_directory(base_dir, date_str))
                else:
                    # VIDEO_DATEが設定されていない場合はベースディレクトリを使用
                    default_dir = get_env_value("DEFAULT_VIDEO_DIR", "./videos/")
                    video_dir = Path(resolve_date_path(default_dir))
            
            if video_dir.exists():
                # 動画ファイル拡張子
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
                video_files = [
                    str(f) for f in video_dir.iterdir() 
                    if f.suffix.lower() in video_extensions
                ]
                
                if video_files:
                    print(f"動画ディレクトリを使用: {video_dir}")
                    return sorted(video_files)
                else:
                    print(f"動画ファイルが見つかりません: {video_dir}")
            else:
                print(f"動画ディレクトリが存在しません: {video_dir}")
        except ValueError as e:
            print(f"日付解析エラー: {e}")
        
        # フォールバック: カメラデバイス0
        print("デフォルトのカメラデバイス0を使用します")
        return 0


def setup_models(lpd_path: str, lpr_path: str, use_gpu: bool) -> tuple[InferenceSession, InferenceSession]:
    """ONNXモデルをセットアップする"""
    providers: List[str] = ["CPUExecutionProvider"]
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    lpd_model = onnxruntime.InferenceSession(lpd_path, providers=providers)
    lpr_model = onnxruntime.InferenceSession(lpr_path, providers=providers)
    
    # ウォームアップ
    _ = run_lpd_inference(lpd_model, np.zeros((960, 540, 3), dtype=np.uint8), 0.3)
    _ = run_lpr_inference(lpr_model, np.zeros((200, 100, 3), dtype=np.uint8))
    
    return lpd_model, lpr_model


def create_video_capture(input_source: Union[int, str], width: int, height: int) -> VideoCapture:
    """VideoCapture オブジェクトを作成する"""
    if isinstance(input_source, int):
        # カメラデバイス
        cap = cv2.VideoCapture(input_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        # ファイルパス
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {input_source}")
    
    return cap


def process_frame(
    frame: np.ndarray,
    lpd_model: InferenceSession,
    lpr_model: InferenceSession,
    lpd_score_th: float,
    lpr_min_width1: int,
    lpr_min_width2: int,
    use_privacy_mode: bool
) -> tuple[np.ndarray, float, float]:
    """単一フレームを処理する"""
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

    return debug_image, lpd_elapsed_time, lpr_elapsed_time


def process_camera(
    device: int,
    models: tuple[InferenceSession, InferenceSession],
    args: argparse.Namespace
) -> None:
    """カメラ入力を処理する"""
    print(f"カメラデバイス {device} を処理中...")
    
    lpd_model, lpr_model = models
    cap = create_video_capture(device, args.width, args.height)
    
    try:
        process_video_stream(cap, models, args, f"Camera {device}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_single_video(
    video_path: str,
    models: tuple[InferenceSession, InferenceSession],
    args: argparse.Namespace,
    output_path: Optional[str] = None
) -> None:
    """単一動画ファイルを処理する"""
    print(f"動画ファイル処理中: {video_path}")
    
    lpd_model, lpr_model = models
    cap = create_video_capture(video_path, args.width, args.height)
    
    try:
        window_title = Path(video_path).name
        if output_path is None:
            output_path = args.output
        
        process_video_stream(cap, models, args, window_title, output_path)
    finally:
        cap.release()


def process_multiple_videos(
    video_paths: List[str],
    models: tuple[InferenceSession, InferenceSession],
    args: argparse.Namespace
) -> None:
    """複数動画ファイルを処理する"""
    print(f"{len(video_paths)} 個の動画ファイルを処理中...")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, video_path in enumerate(video_paths):
        print(f"進捗: {i+1}/{len(video_paths)} - {Path(video_path).name}")
        
        # 出力ファイル名を生成
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"{video_name}_processed.avi")
        
        try:
            process_single_video(video_path, models, args, output_path)
        except Exception as e:
            print(f"エラー（{video_path}）: {e}")
            continue
    
    print("全ての動画処理が完了しました。")


def process_image(
    image_path: str,
    models: tuple[InferenceSession, InferenceSession],
    args: argparse.Namespace
) -> None:
    """単一画像を処理する"""
    print(f"画像ファイル処理中: {image_path}")
    
    lpd_model, lpr_model = models
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError(f"画像ファイルを読み込めません: {image_path}")
    
    debug_image, lpd_time, lpr_time = process_frame(
        frame, lpd_model, lpr_model,
        args.lpd_score_th, args.lpr_min_width1, args.lpr_min_width2,
        args.use_privacy_mode
    )
    
    # ヘッドレスモードでない場合のみGUI表示
    if not args.headless:
        cv2.imshow(f"Image: {Path(image_path).name}", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # ヘッドレスモードの場合は結果を保存
        output_path = f"processed_{Path(image_path).name}"
        cv2.imwrite(output_path, debug_image)
        print(f"結果を保存: {output_path}")


def process_video_stream(
    cap: VideoCapture,
    models: tuple[InferenceSession, InferenceSession],
    args: argparse.Namespace,
    window_title: str,
    output_path: Optional[str] = None
) -> None:
    """動画ストリームを処理する"""
    lpd_model, lpr_model = models
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer: Optional[cv2.VideoWriter] = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            debug_image, lpd_time, lpr_time = process_frame(
                frame, lpd_model, lpr_model,
                args.lpd_score_th, args.lpr_min_width1, args.lpr_min_width2,
                args.use_privacy_mode
            )
            
            # ヘッドレスモードでない場合のみGUI表示
            if not args.headless:
                cv2.imshow(window_title, debug_image)
                
                # ESCキーで終了（カメラの場合のみ）
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            else:
                # ヘッドレスモードの場合、進捗表示
                if frame_count % 30 == 0:  # 30フレームごとに表示
                    print(f"処理中... フレーム: {frame_count}")
            
            # 動画書き込み
            if args.use_video_writer and output_path:
                if video_writer is None:
                    debug_height, debug_width = debug_image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        int(cap_fps),
                        (debug_width, debug_height),
                    )
                if video_writer:
                    video_writer.write(debug_image)
    
    finally:
        if video_writer:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main() -> None:
    """メイン処理"""
    args = get_args()
    
    # 入力タイプを決定
    try:
        video_input: VideoInput = determine_input_type(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
        return
    
    # モデルセットアップ
    try:
        models = setup_models(args.lpd, args.lpr, args.use_gpu)
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return
    
    # 入力タイプに応じて処理を分岐
    try:
        if isinstance(video_input, int):
            # カメラデバイス
            process_camera(video_input, models, args)
        elif isinstance(video_input, str):
            # 単一ファイル（動画 or 画像）
            if args.image is not None:
                process_image(video_input, models, args)
            else:
                process_single_video(video_input, models, args)
        elif isinstance(video_input, list):
            # 複数動画ファイル
            process_multiple_videos(video_input, models, args)
        else:
            raise TypeError(f"サポートされていない入力タイプ: {type(video_input)}")
    
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"処理エラー: {e}")
    finally:
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