from typing import Tuple, List, Any

import cv2
import onnxruntime  # type:ignore
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 地域名ID
region_dict = {
    "札幌": 0,
    "函館": 1,
    "旭川": 2,
    "室蘭": 3,
    "苫小牧": 4,
    "釧路": 5,
    "帯広": 7,
    "北見": 8,
    "知床": 9,
    "青森": 10,
    "弘前": 11,
    "八戸": 12,
    "岩手": 13,
    "盛岡": 14,
    "平泉": 15,
    "宮城": 16,
    "仙台": 17,
    "秋田": 18,
    "山形": 19,
    "庄内": 20,
    "福島": 21,
    "会津": 22,
    "郡山": 23,
    "白河": 24,
    "いわき": 25,
    "水戸": 26,
    "土浦": 27,
    "つくば": 28,
    "宇都宮": 29,
    "那須": 30,
    "とちぎ": 31,
    "群馬": 32,
    "前橋": 33,
    "高崎": 34,
    "大宮": 35,
    "川口": 36,
    "所沢": 37,
    "川越": 38,
    "熊谷": 39,
    "春日部": 40,
    "越谷": 41,
    "千葉": 42,
    "成田": 43,
    "習志野": 44,
    "市川": 45,
    "船橋": 46,
    "袖ヶ浦": 47,
    "市原": 48,
    "野田": 49,
    "柏": 50,
    "松戸": 51,
    "品川": 52,
    "世田谷": 53,
    "練馬": 54,
    "杉並": 55,
    "板橋": 56,
    "足立": 57,
    "江東": 58,
    "葛飾": 59,
    "八王子": 60,
    "多摩": 61,
    "横浜": 62,
    "川崎": 63,
    "湘南": 64,
    "相模": 65,
    "山梨": 66,
    "富士山": 67,
    "新潟": 68,
    "長岡": 69,
    "上越": 70,
    "長野": 71,
    "松本": 72,
    "諏訪": 73,
    "富山": 74,
    "石川": 75,
    "金沢": 76,
    "福井": 77,
    "岐阜": 78,
    "飛騨": 79,
    "静岡": 80,
    "浜松": 81,
    "沼津": 82,
    "伊豆": 83,
    "名古屋": 84,
    "豊橋": 85,
    "三河": 86,
    "岡崎": 87,
    "豊田": 88,
    "尾張小牧": 89,
    "一宮": 90,
    "春日井": 91,
    "三重": 92,
    "鈴鹿": 93,
    "四日市": 94,
    "伊勢志摩": 95,
    "滋賀": 96,
    "京都": 97,
    "なにわ": 98,
    "大阪": 99,
    "和泉": 100,
    "堺": 101,
    "奈良": 102,
    "飛鳥": 103,
    "和歌山": 104,
    "神戸": 105,
    "姫路": 106,
    "鳥取": 107,
    "島根": 108,
    "出雲": 109,
    "岡山": 110,
    "倉敷": 111,
    "広島": 112,
    "福山": 113,
    "山口": 114,
    "下関": 115,
    "徳島": 116,
    "香川": 117,
    "高松": 118,
    "愛媛": 119,
    "高知": 120,
    "福岡": 121,
    "北九州": 122,
    "久留米": 123,
    "筑豊": 124,
    "佐賀": 125,
    "長崎": 126,
    "佐世保": 127,
    "熊本": 128,
    "大分": 129,
    "宮崎": 130,
    "鹿児島": 131,
    "奄美": 132,
    "沖縄": 133,
    "十勝": 134,
    "日光": 135,
    "江戸川": 136,
    "安曇野": 137,
    "南信州": 138,
}

# ひらがなID
hiragana_dict = {
    "あ": 0,
    "い": 1,
    "う": 2,
    "え": 3,
    "か": 4,
    "き": 5,
    "く": 6,
    "け": 7,
    "こ": 8,
    "さ": 9,
    "す": 10,
    "そ": 11,
    "せ": 12,
    "た": 13,
    "ち": 14,
    "つ": 15,
    "て": 16,
    "と": 17,
    "な": 18,
    "に": 19,
    "ぬ": 20,
    "ね": 21,
    "の": 22,
    "は": 23,
    "ひ": 24,
    "ふ": 25,
    "ほ": 26,
    "ま": 27,
    "み": 28,
    "む": 29,
    "め": 30,
    "も": 31,
    "や": 32,
    "ゆ": 33,
    "よ": 34,
    "ら": 35,
    "り": 36,
    "る": 37,
    "れ": 38,
    "ろ": 39,
    "わ": 40,
    "を": 41,
    "A": 42,
    "B": 43,
    "C": 44,
    "E": 45,
    "H": 46,
    "K": 47,
    "L": 48,
    "M": 49,
    "T": 50,
    "Y": 51,
    "V": 52,
}

# 分類番号ID（1桁目）
class_num_01_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
}

# 分類番号ID（2桁目）
class_num_02_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "C": 11,
    "F": 12,
    "H": 13,
    "K": 14,
    "L": 15,
    "M": 16,
    "P": 17,
    "X": 18,
    "Y": 19,
}

# 分類番号ID（3桁目）
class_num_03_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "C": 11,
    "F": 12,
    "H": 13,
    "K": 14,
    "L": 15,
    "M": 16,
    "P": 17,
    "X": 18,
    "Y": 19,
    " ": 20,
}


# LPD推論用関数
def run_lpd_inference(
    onnx_session: onnxruntime.InferenceSession, image: np.ndarray, score_th: float
) -> np.ndarray:
    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape

    # 前処理
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize_image = cv2.resize(bgr_image, (input_shape[3], input_shape[2]))
    input_image = resize_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # ONNX推論
    result = onnx_session.run(None, {input_name: input_image})

    # 後処理
    detection_results = result[0][0]
    detection_results = detection_results[detection_results[:, 4] >= score_th]

    # x1, y1, x2, y2正規化
    if detection_results.shape[0] > 0:
        detection_results[:, 0] /= input_shape[3]  # x1
        detection_results[:, 1] /= input_shape[2]  # y1
        detection_results[:, 2] /= input_shape[3]  # x2
        detection_results[:, 3] /= input_shape[2]  # y2

    return detection_results


def run_lpr_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> Tuple[int, int, List[int], List[int]]:
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # 前処理
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image_f32 = input_image.astype("float32")
    input_image_f32 /= 255.0

    # ONNX推論
    input_name = onnx_session.get_inputs()[0].name
    onnx_result = onnx_session.run(
        [
            "region_id",
            "hiragana_id",
            "class_num_01",
            "class_num_02",
            "class_num_03",
            "plate_num_01",
            "plate_num_02",
            "plate_num_03",
            "plate_num_04",
        ],
        {input_name: input_image_f32},
    )

    # 後処理
    region_id: int = int(np.argmax(np.squeeze(onnx_result[0])))
    hiragana_id: int = int(np.argmax(np.squeeze(onnx_result[1])))
    class_num_ids: List[int] = [
        int(np.argmax(np.squeeze(onnx_result[2]))),
        int(np.argmax(np.squeeze(onnx_result[3]))),
        int(np.argmax(np.squeeze(onnx_result[4]))),
    ]
    plate_num_ids: List[int] = [
        int(np.argmax(np.squeeze(onnx_result[5]))),
        int(np.argmax(np.squeeze(onnx_result[6]))),
        int(np.argmax(np.squeeze(onnx_result[7]))),
        int(np.argmax(np.squeeze(onnx_result[8]))),
    ]

    return hiragana_id, region_id, class_num_ids, plate_num_ids


# 日本語描画
class CvDrawText:
    def __init__(self) -> None:
        pass

    @classmethod
    def puttext(
        cls,
        cv_image: np.ndarray,
        text: str,
        point: Tuple[int, int],
        font_path: str,
        font_size: int,
        color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        font = ImageFont.truetype(font_path, font_size)

        cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)

        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)

        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

        return cv_bgr_result_image
