> [!CAUTION]
> * ナンバープレート検出とナンバープレート認識のデータセットは非公開です
> * ナンバープレート検出用のデータセットは、作成者(高橋)の自宅周辺を中心に撮影して収集しています<br>推論時の背景や車種、道路種別によっては著しく検出率が悪化する可能性があります
> * ナンバープレート認識用のデータセットは、Google画像検索にて収集しています<br>推論時の撮影角度や昼夜などの映り方によっては著しく認識率が悪化する可能性があります

# PlateYOLO-JP-Prototype
日本のナンバープレート検出と認識の技術検証用プロトタイプです。

<img src="https://github.com/user-attachments/assets/441efae7-5d96-41b6-8a69-4a21bb0a5fc1" loading="lazy" width="45%"> <img src="https://github.com/user-attachments/assets/a4ab69a5-91e2-41dc-9a08-b3bef89bda42" loading="lazy" width="45%">

# Pipeline
<img src="https://github.com/user-attachments/assets/a893b940-91c7-4aa7-82e7-1378f28e4299" loading="lazy" width="95%"><br>
ナンバープレート検出とナンバープレート認識の2ステージ構成です。
* PlateYOLO-JP：YOLO12ベースの検出器
* EkMixier：マルチカーネル構造とECAブロックを持つパッチエンベディング系のクラス分類器<br>※対応地域名や対応ひらがな等は、util.py の region_dict や hiragana_dict を参照ください

> [!NOTE]
> * PlateYOLO-JP は、ある程度距離が離れた車両のナンバープレート検出を想定しています<br>画面全体にナンバープレートを映した画像などは検出できません

# Requirements
```
onnx                 1.18.0    or later
onnxruntime          1.18.0    or later
opencv-python        4.11.0.86 or later
pillow               11.2.1 or later
```

# Usage
アプリの起動方法は以下です。
```bash
python demo.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラデバイスのキャプチャ幅<br>
デフォルト：960
* --height<br>
カメラデバイスのキャプチャ高さ<br>
デフォルト：540
* --lpd<br>
ナンバープレート検出モデル<br>
デフォルト：weight/PlateYOLO-JP-640x640.onnx
* --lpr<br>
ナンバープレート認識モデル<br>
デフォルト：weight/EkMixer-128x128.onnx
* --lpd_score_th<br>
ナンバープレート検出閾値<br>
デフォルト：0.3
* --lpr_min_width1<br>
ナンバープレート認識最小幅1：この最小幅を下回るナンバープレートは「認識不可」と判定<br>
デフォルト：110
* --lpr_min_width2<br>
ナンバープレート認識最小幅2：この最小幅を下回るナンバープレートの地域名、分類番号、ひらがなは「認識不可」と判定<br>
デフォルト：150
* --use_video_writer<br>
動画書き込み<br>
デフォルト：指定なし
* --output<br>
動画書き込み時のファイルパス<br>
デフォルト：output.avi
* --unuse_gpu<br>
GPU推論なし（CPU推論）<br>
デフォルト：指定なし
* --use_privacy_mode<br>
プライバシー表示<br>
デフォルト：指定なし

# Font
* [源泉丸ゴシックフォント](https://github.com/ButTaiwan/gensen-font)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
PlateYOLO-JP-Prototype is under [AGPL license](LICENSE).<br>
PlateYOLO-JP-Prototype は [AGPL license](LICENSE)ですが、源泉丸ゴシックフォントは [SIL Open Font License 1.1](font/gensen-font/SIL_Open_Font_License_1.1.txt) です。
