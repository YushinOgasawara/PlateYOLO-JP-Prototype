import os

# =============================
# 画像検出関連設定
# =============================

# 対象ディレクトリパス（日付フォルダが含まれる親ディレクトリ）
PICTURE_IMAGE_DIR = "/path/to/your/image/directory"

# 検出された画像を保存するinputsディレクトリのパス
INPUTS_DIR = "inputs"

# =============================
# 物体検出設定
# =============================

# FPS監視の有効/無効
PICTURE_ENABLE_FPS_MONITORING = True

# 処理間隔（何枚ごとに処理するか）
PICTURE_PROCESS_INTERVAL = 3

# FPS計算間隔（枚数）
PICTURE_FPS_INTERVAL = 100

# =============================
# 終了条件設定
# =============================

# 終了時刻（時）
PICTURE_EXIT_HOUR = 23

# 終了時刻（分）
PICTURE_EXIT_MINUTE = 59

# 最大実行時間（時間）
PICTURE_MAX_RUNTIME_HOURS = 24

# =============================
# ディレクトリ作成
# =============================

def ensure_directories():
    """必要なディレクトリを作成"""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    
def get_date_from_path(image_path):
    """
    画像パスから日付情報を取得
    
    Args:
        image_path (str): 画像ファイルのパス
    
    Returns:
        str: 日付文字列（YYYY-MM-DD形式）、取得できない場合はNone
    """
    try:
        # パスを分割して日付フォルダを探す
        path_parts = image_path.split(os.sep)
        for part in path_parts:
            # YYYY-MM-DD形式の文字列を探す
            if len(part) == 10 and part.count('-') == 2:
                year, month, day = part.split('-')
                if (len(year) == 4 and year.isdigit() and 
                    len(month) == 2 and month.isdigit() and 
                    len(day) == 2 and day.isdigit()):
                    return part
        return None
    except Exception:
        return None

def get_inputs_date_dir(date_str):
    """
    指定された日付のinputsディレクトリパスを取得
    
    Args:
        date_str (str): 日付文字列（YYYY-MM-DD形式）
    
    Returns:
        str: inputsディレクトリ内の日付ディレクトリパス
    """
    return os.path.join(INPUTS_DIR, date_str)