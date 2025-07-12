# PlateYOLO-JP-Prototype システム仕様書

## 1. システム概要

### 1.1 プロジェクト名
PlateYOLO-JP-Prototype

### 1.2 システム目的
日本のナンバープレート検出・認識のリアルタイム技術検証用プロトタイプシステム

### 1.3 システム概要
本システムは、カメラ映像またはファイル（動画/画像）から日本の自動車ナンバープレートを検出し、文字認識を行う2段階AI推論システムです。

## 2. システム構成

### 2.1 ディレクトリ構造
```
PlateYOLO-JP-Prototype/
├── demo.py              # メインアプリケーション
├── main.py              # シンプルなHello World実装
├── util.py              # ユーティリティ関数・辞書定義
├── pyproject.toml       # プロジェクト設定ファイル
├── requirements.txt     # 依存関係
├── uv.lock             # 依存関係ロックファイル
├── weight/             # 学習済みモデル
│   ├── PlateYOLO-JP-320x320.onnx
│   ├── PlateYOLO-JP-640x640.onnx    # デフォルト検出モデル
│   ├── PlateYOLO-JP-1280x1280.onnx
│   ├── PlateYOLO-JP-1920x1920.onnx
│   └── EkMixer-128x128.onnx         # 文字認識モデル
├── font/               # フォントファイル
│   └── gensen-font/
└── LICENSE             # AGPLライセンス
```

### 2.2 主要ファイル

#### 2.2.1 demo.py
- **行数**: 361行
- **機能**: メインアプリケーション
- **主要処理**:
  - コマンドライン引数解析
  - カメラ/動画/画像入力処理
  - 2段階AI推論（検出→認識）
  - リアルタイム結果表示
  - 動画出力機能

#### 2.2.2 util.py
- **行数**: 379行
- **機能**: ユーティリティ関数・辞書定義
- **主要処理**:
  - 地域名辞書（149地域対応）
  - ひらがな辞書（53文字対応）
  - 分類番号辞書（3桁対応）
  - ONNX推論関数
  - 日本語文字描画クラス

#### 2.2.3 main.py
- **行数**: 7行
- **機能**: シンプルなHello Worldプログラム

## 3. 技術仕様

### 3.1 開発環境
- **言語**: Python 3.12+
- **パッケージ管理**: uv
- **AI推論**: ONNX Runtime

### 3.2 依存ライブラリ
```
onnx==1.18.0
onnxruntime==1.18.0
opencv-python==4.11.0.86
pillow==11.2.1
numpy<2  # NumPy 2.x互換性対応
```

### 3.3 AIモデル構成

#### 3.3.1 検出モデル: PlateYOLO-JP
- **ベース**: YOLO12
- **入力サイズ**: 320x320, 640x640, 1280x1280, 1920x1920
- **出力**: バウンディングボックス座標 + 信頼度スコア
- **用途**: ナンバープレートの位置検出

#### 3.3.2 認識モデル: EkMixer
- **構造**: ECAブロック + マルチカーネル構造パッチエンベディング
- **入力サイズ**: 128x128
- **出力**: 地域名、ひらがな、分類番号、一連指定番号
- **用途**: ナンバープレート文字認識

## 4. 機能仕様

### 4.1 入力オプション
| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| --device | 0 | カメラデバイス番号 |
| --video | None | 動画ファイルパス |
| --image | None | 画像ファイルパス |
| --width | 960 | キャプチャ幅 |
| --height | 540 | キャプチャ高さ |

### 4.2 モデル設定
| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| --lpd | weight/PlateYOLO-JP-640x640.onnx | 検出モデル |
| --lpr | weight/EkMixer-128x128.onnx | 認識モデル |
| --lpd_score_th | 0.3 | 検出信頼度閾値 |

### 4.3 認識品質制御
| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| --lpr_min_width1 | 110 | 最小認識幅1（下回ると「認識不可」） |
| --lpr_min_width2 | 150 | 最小認識幅2（地域名等制限） |

### 4.4 出力オプション
| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| --use_video_writer | False | 動画保存有効化 |
| --output | output.avi | 出力動画ファイル名 |
| --use_gpu | False | GPU推論有効化 |
| --use_privacy_mode | False | プライバシーモード |

## 5. 処理フロー

### 5.1 システム起動フロー
1. コマンドライン引数解析
2. ONNX推論セッション初期化
3. VideoCapture初期化
4. ウォームアップ推論実行

### 5.2 リアルタイム処理フロー
1. フレーム取得（カメラ/動画/画像）
2. **ステップ1**: ナンバープレート検出
   - 画像前処理（リサイズ、正規化）
   - PlateYOLO-JP推論実行
   - バウンディングボックス後処理
3. **ステップ2**: 文字認識（検出された各プレートに対して）
   - プレート領域切り出し
   - EkMixer推論実行
   - 文字ID→文字変換
4. 結果描画・表示
5. 動画保存（オプション）

### 5.3 推論詳細

#### 5.3.1 検出推論（run_lpd_inference関数）
- BGR→RGB色空間変換
- モデル入力サイズにリサイズ
- CHW形式変換・正規化
- ONNX推論実行
- 信頼度フィルタリング
- 座標正規化

#### 5.3.2 認識推論（run_lpr_inference関数）
- 128x128リサイズ
- BGR→RGB色空間変換
- CHW形式変換・正規化
- 9出力ヘッド推論実行
- argmax後処理

## 6. 対応文字・地域

### 6.1 対応地域（149地域）
北海道から沖縄まで全国の陸運局・自動車検査登録事務所管轄地域

### 6.2 対応ひらがな（53文字）
あ〜ろ、わ、を + A、B、C、E、H、K、L、M、T、Y、V

### 6.3 分類番号
- 1桁目: 0-9
- 2桁目: 0-9、A、C、F、H、K、L、M、P、X、Y
- 3桁目: 0-9、A、C、F、H、K、L、M、P、X、Y、スペース

## 7. パフォーマンス

### 7.1 表示情報
- 総処理時間（LPD + LPR）
- LPD処理時間
- LPR処理時間（検出数、平均時間）

### 7.2 プライバシー機能
- プレート領域のマスキング
- 分類番号・一連指定番号の一部秘匿

## 8. 制限事項

### 8.1 データセット制限
- 検出用データセット: 作成者自宅周辺中心
- 認識用データセット: Google画像検索収集
- 背景・車種・撮影条件により性能変動

### 8.2 技術制限
- 近距離プレート検出不可（画面全体プレートなど）
- 撮影角度・昼夜条件により認識率変動
- NumPy 2.x互換性問題（numpy<2で回避）

## 9. ライセンス
- **メインシステム**: AGPL License
- **フォント**: SIL Open Font License 1.1

## 10. モジュール詳細解説

### 10.1 demo.py 詳細解説

#### 10.1.1 ファイル概要
- **ファイル名**: demo.py
- **総行数**: 361行
- **エンコーディング**: UTF-8
- **実行方式**: スクリプト直接実行またはuvコマンド実行

#### 10.1.2 インポート構成 (1-20行)
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy        # オブジェクトの深いコピー用
import time        # 処理時間計測用
import argparse    # コマンドライン引数解析用

import cv2         # OpenCV - 画像処理・カメラ操作
import numpy as np # NumPy - 数値計算・配列操作
import onnxruntime # ONNX Runtime - AI推論エンジン

from util import (  # 自作ユーティリティモジュール
    region_dict, hiragana_dict,           # 地域・ひらがな辞書
    class_num_01_dict, class_num_02_dict, class_num_03_dict,  # 分類番号辞書
    run_lpd_inference, run_lpr_inference,  # AI推論関数
    CvDrawText,                           # 日本語描画クラス
)
```

#### 10.1.3 引数解析関数: get_args() (23-59行)
**機能**: コマンドライン引数の定義・解析
**戻り値**: argparse.Namespace

**引数詳細**:
- **入力系**:
  - `--device`: カメラデバイス番号 (default: 0)
  - `--video`: 動画ファイルパス (default: None)
  - `--image`: 画像ファイルパス (default: None)
  - `--width/--height`: キャプチャサイズ (default: 960x540)

- **モデル系**:
  - `--lpd`: 検出モデルパス (default: "weight/PlateYOLO-JP-640x640.onnx")
  - `--lpr`: 認識モデルパス (default: "weight/EkMixer-128x128.onnx")

- **品質制御系**:
  - `--lpd_score_th`: 検出信頼度閾値 (default: 0.3)
  - `--lpr_min_width1/2`: 認識最小幅閾値 (default: 110/150)

- **出力・動作系**:
  - `--use_video_writer`: 動画保存フラグ
  - `--output`: 出力ファイル名 (default: "output.avi")
  - `--use_gpu`: GPU推論フラグ
  - `--use_privacy_mode`: プライバシーモードフラグ

#### 10.1.4 メイン処理関数: main() (62-200行)

**フェーズ1: 初期化処理 (62-93行)**
```python
# 引数取得・変数設定
args = get_args()
device, cap_width, cap_height = args.device, args.width, args.height

# 実行プロバイダー設定
providers = ["CPUExecutionProvider"]
if args.use_gpu:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ONNXモデル読み込み
lpd_model = onnxruntime.InferenceSession(lpd_model_path, providers=providers)
lpr_model = onnxruntime.InferenceSession(lpr_model_path, providers=providers)
```

**フェーズ2: VideoCapture初期化 (94-102行)**
- カメラ/動画ファイルの選択的初期化
- キャプチャプロパティ設定
- FPS取得

**フェーズ3: ウォームアップ (106-108行)**
```python
# ダミーデータでのウォームアップ推論
_ = run_lpd_inference(lpd_model, np.zeros((960, 540, 3), dtype=np.uint8), 0.3)
_ = run_lpr_inference(lpr_model, np.zeros((200, 100, 3), dtype=np.uint8))
```

**フェーズ4: メインループ (110-200行)**

**ステップ1: フレーム取得 (111-117行)**
```python
if image_path is None:
    ret, frame = cap.read()  # カメラ/動画から取得
else:
    frame = cv2.imread(image_path)  # 画像ファイルから取得
```

**ステップ2: ナンバープレート検出 (119-123行)**
```python
lpd_start_time = time.perf_counter()
detection_results = run_lpd_inference(lpd_model, frame, lpd_score_th)
lpd_end_time = time.perf_counter()
lpd_elapsed_time = (lpd_end_time - lpd_start_time) * 1000  # ms変換
```

**ステップ3: 文字認識ループ (125-158行)**
```python
lpr_results = []
for detection_result in detection_results:
    # バウンディングボックス座標計算
    x1 = int(detection_result[0] * frame_width) - offset
    y1 = int(detection_result[1] * frame_height) - offset
    x2 = int(detection_result[2] * frame_width) + offset
    y2 = int(detection_result[3] * frame_height) + offset
    
    # プレート領域切り出し
    lp_image = frame[y1:y2, x1:x2]
    
    # 認識推論実行
    hiragana_id, region_id, class_num_ids, plate_num_ids = run_lpr_inference(lpr_model, lp_image)
    
    # 結果辞書作成
    lpr_results.append({
        "bbox": detection_result[:4],
        "bbox_score": detection_result[4],
        "bbox_class_id": detection_result[5],
        "lp_shape": lp_image.shape,
        "hiragana_id": hiragana_id,
        "region_id": region_id,
        "class_num_ids": class_num_ids,
        "plate_num_ids": plate_num_ids,
    })
```

**ステップ4: 表示・保存処理 (160-200行)**
- `draw_info()`関数による描画
- OpenCVウィンドウ表示
- 動画書き込み（オプション）

#### 10.1.5 描画関数: draw_info() (203-356行)

**機能**: 検出・認識結果の可視化
**引数**: 
- `image`: 元画像
- `lpr_results`: 認識結果リスト
- `lpd_elapsed_time, lpr_elapsed_time`: 処理時間
- 各種辞書・設定値

**処理詳細**:

**フェーズ1: 画像準備・辞書反転 (219-233行)**
```python
debug_image = copy.deepcopy(image)
debug_image = cv2.resize(debug_image, (resize_width, ...))  # 960px幅リサイズ

# ID→文字変換用の逆引き辞書作成
region_dict_inv = {v: k for k, v in region_dict.items()}
hiragana_dict_inv = {v: k for k, v in hiragana_dict.items()}
```

**フェーズ2: ナンバープレート描画ループ (235-326行)**
```python
for lpr_result in lpr_results:
    # バウンディングボックス描画
    x1, y1, x2, y2 = 座標計算
    if not use_privacy_mode:
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 枠線
    else:
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), -1)  # 塗りつぶし
    
    # 文字認識結果の品質判定・テキスト生成
    lp_width = lpr_result["lp_shape"][1]
    
    # 地域名処理
    if lpr_min_width1 > lp_width:
        region_text = ""
    elif lpr_min_width2 > lp_width:
        region_text = "読取不可"
    else:
        region_text = region_dict_inv.get(lpr_result["region_id"], 0)
    
    # 分類番号処理（プライバシーモード対応）
    class_text = ""
    if not use_privacy_mode:
        class_text += class_num_02_dict_inv[lpr_result["class_num_ids"][1]]
    else:
        class_text += "X"  # マスク文字
```

**フェーズ3: 日本語テキスト描画 (311-326行)**
```python
debug_image = CvDrawText.puttext(
    debug_image,
    region_text + " " + class_text,      # 上段：地域名+分類番号
    (x1, y1 - 30 - 2),
    font_path,
    15,
    (0, 255, 0),
)
debug_image = CvDrawText.puttext(
    debug_image,
    hiragana_text + " " + plate_text,    # 下段：ひらがな+一連番号
    (x1, y1 - 15 - 2),
    font_path,
    15,
    (0, 255, 0),
)
```

**フェーズ4: パフォーマンス情報描画 (328-354行)**
```python
lpr_count = len(lpr_results)
lpr_mean_time = (lpr_elapsed_time / lpr_count) if lpr_count > 0 else 0

# 3行のパフォーマンス表示
debug_image = CvDrawText.puttext(debug_image, f"Total:{total_time:.0f}ms", ...)
debug_image = CvDrawText.puttext(debug_image, f"    LPD:{lpd_time:.0f}ms", ...)
debug_image = CvDrawText.puttext(debug_image, f"    LPR:{lpr_time:.0f}ms(count:{count}, avg:{avg:.0f}ms)", ...)
```

#### 10.1.6 エントリーポイント (359-361行)
```python
if __name__ == "__main__":
    main()
```

### 10.2 util.py 詳細解説

#### 10.2.1 ファイル概要
- **ファイル名**: util.py
- **総行数**: 379行
- **機能**: AI推論・文字変換・描画ユーティリティ

#### 10.2.2 インポート構成 (1-6行)
```python
from typing import Tuple, List, Any    # 型ヒント
import cv2                             # 画像処理
import onnxruntime                     # AI推論
import numpy as np                     # 数値計算
from PIL import ImageFont, ImageDraw, Image  # 日本語描画用PIL
```

#### 10.2.3 辞書定義セクション (8-268行)

**地域名辞書 (9-148行)**
- **要素数**: 149地域
- **構造**: `"地域名": ID` の辞書形式
- **範囲**: 全国の陸運局・自動車検査登録事務所管轄
- **例**: `"札幌": 0, "函館": 1, ..., "沖縄": 133`

**ひらがな辞書 (151-205行)**
- **要素数**: 53文字
- **構造**: `"文字": ID` の辞書形式
- **範囲**: あ〜ろ + わ、を + アルファベット特定文字
- **例**: `"あ": 0, "い": 1, ..., "V": 52`

**分類番号辞書 (207-268行)**
- **class_num_01_dict (1桁目)**: 0-9 (10要素)
- **class_num_02_dict (2桁目)**: 0-9 + A,C,F,H,K,L,M,P,X,Y (20要素)
- **class_num_03_dict (3桁目)**: 2桁目 + スペース (21要素)

#### 10.2.4 LPD推論関数: run_lpd_inference() (271-298行)

**機能**: ナンバープレート検出推論
**引数**:
- `onnx_session`: ONNXInferenceSession
- `image`: 入力画像 (np.ndarray)
- `score_th`: 信頼度閾値 (float)
**戻り値**: 検出結果配列 (np.ndarray)

**処理フロー**:
```python
# 1. 入力仕様取得
input_name = onnx_session.get_inputs()[0].name
input_shape = onnx_session.get_inputs()[0].shape  # [batch, channel, height, width]

# 2. 前処理
bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # BGR→RGB変換
resize_image = cv2.resize(bgr_image, (input_shape[3], input_shape[2]))  # リサイズ
input_image = resize_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC→CHW, 正規化
input_image = np.expand_dims(input_image, axis=0)               # バッチ次元追加

# 3. ONNX推論実行
result = onnx_session.run(None, {input_name: input_image})

# 4. 後処理
detection_results = result[0][0]                                # バッチ次元削除
detection_results = detection_results[detection_results[:, 4] >= score_th]  # 信頼度フィルタ

# 5. 座標正規化 (モデル入力サイズ→0-1座標)
detection_results[:, 0] /= input_shape[3]  # x1正規化
detection_results[:, 1] /= input_shape[2]  # y1正規化
detection_results[:, 2] /= input_shape[3]  # x2正規化
detection_results[:, 3] /= input_shape[2]  # y2正規化
```

#### 10.2.5 LPR推論関数: run_lpr_inference() (301-349行)

**機能**: ナンバープレート文字認識推論
**引数**:
- `onnx_session`: ONNXInferenceSession
- `image`: プレート画像 (np.ndarray)
**戻り値**: (hiragana_id, region_id, class_num_ids, plate_num_ids)

**処理フロー**:
```python
# 1. 入力仕様取得
input_size = onnx_session.get_inputs()[0].shape
input_width, input_height = input_size[3], input_size[2]  # 通常128x128

# 2. 前処理
input_image = cv2.resize(image, dsize=(input_width, input_height))    # 128x128リサイズ
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)            # BGR→RGB
input_image = input_image.transpose(2, 0, 1)                         # HWC→CHW
input_image = np.expand_dims(input_image, axis=0)                     # バッチ次元
input_image_f32 = input_image.astype("float32") / 255.0               # float32正規化

# 3. マルチヘッド推論実行
input_name = onnx_session.get_inputs()[0].name
onnx_result = onnx_session.run([
    "region_id",      # 地域名出力
    "hiragana_id",    # ひらがな出力
    "class_num_01",   # 分類番号1桁目
    "class_num_02",   # 分類番号2桁目
    "class_num_03",   # 分類番号3桁目
    "plate_num_01",   # 一連番号1桁目
    "plate_num_02",   # 一連番号2桁目
    "plate_num_03",   # 一連番号3桁目
    "plate_num_04",   # 一連番号4桁目
], {input_name: input_image_f32})

# 4. 後処理（各出力のargmax）
region_id = int(np.argmax(np.squeeze(onnx_result[0])))
hiragana_id = int(np.argmax(np.squeeze(onnx_result[1])))
class_num_ids = [int(np.argmax(np.squeeze(onnx_result[i]))) for i in range(2, 5)]
plate_num_ids = [int(np.argmax(np.squeeze(onnx_result[i]))) for i in range(5, 9)]
```

#### 10.2.6 日本語描画クラス: CvDrawText (353-379行)

**機能**: OpenCV画像への日本語テキスト描画
**実装方式**: PIL (Python Imaging Library) 経由描画

**クラスメソッド: puttext()**
```python
@classmethod
def puttext(cls, cv_image, text, point, font_path, font_size, color=(0,0,0)):
    # 1. TrueTypeフォント読み込み
    font = ImageFont.truetype(font_path, font_size)
    
    # 2. OpenCV→PIL変換
    cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_rgb_image)
    
    # 3. PIL描画実行
    draw = ImageDraw.Draw(pil_image)
    draw.text(point, text, fill=color, font=font)
    
    # 4. PIL→OpenCV変換
    cv_rgb_result_image = np.asarray(pil_image)
    cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)
    
    return cv_bgr_result_image
```

**特徴**:
- OpenCVの制限（日本語フォント非対応）を回避
- PIL経由での高品質日本語描画
- TrueTypeフォント（源泉丸ゴシック）対応
