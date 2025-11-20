from ultralytics import YOLO
import cv2
import sys
import os

# 親ディレクトリのパスを追加してsave_on_device/configをimport
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

class Detector:
    """
    フレームからのYOLOによる車体検出を行うクラス
    """
    def __init__(self):
        """
        コンストラクタ
        - self.model: YOLOv8モデルのロード
        - self.running: 物体検出の実行を管理
        - self.tracked_ids: 追跡中のIDを管理
        """
        self.model = YOLO(config.YOLO_MODEL)
        self.running = False
        self.tracked_ids = set()  # 追跡中のIDを管理
    
    def detect_object(self, ret, frame):
        """
        YOLOでフレーム解析して車体検出
        
        Args:
            ret: bool値
            frame: 画像データ
        
        Returns:
            tuple: (is_detected, new_ids, frame)
                - is_detected: 車体検出フラグ
                - new_ids: 新しく検出されたIDのリスト
                - frame: 検出されたフレーム
        """
        
        if not self.running:
            self.running = True
        
        # カメラからフレームを取得
        ret, frame = ret, frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # print("frame",frame.shape)
        
        #フレーム取得に失敗した場合
        if not ret or frame is None:
                return False, [], None
        
        # YOLOv8モデルを使用して車体検出
        # クラス2は'car'を表す（YOLOv8のCOCOデータセット）
        results = self.model.track(

            frame, 
            classes=[config.CAR_CLASS_ID], 
            persist=True,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,  # 信頼度閾値
            iou=config.YOLO_IOU_THRESHOLD,          # IoU閾値
            verbose=False  # 詳細ログを無効化
        )

        
        # 検出結果を確認
        if len(results) > 0:
            boxes = results[0].boxes
            
            # 車体が検出されたかチェック
            if len(boxes) > 0 and boxes.cls.numel() > 0:

                # 検出された物体の信頼度をログ出力（デバッグ用）
                if hasattr(boxes, 'conf') and boxes.conf is not None:
                    confidences = boxes.conf.cpu().numpy()
                    print(f"検出された物体の信頼度: {confidences}")
                
                # ID情報を取得
                if hasattr(boxes, 'id') and boxes.id is not None:
                    current_ids = set(boxes.id.cpu().numpy().astype(int))
                    # 新しいIDを検出
                    new_ids = current_ids - self.tracked_ids
                    # 追跡中のIDを更新
                    self.tracked_ids = current_ids
                    
                    if new_ids:
                        print(f"新しいID検出: {new_ids}, 現在追跡中: {self.tracked_ids}")
                    
                    return True, list(new_ids), frame
        
        return False, [], frame
    
    def stop(self):
        """
        YOLO車体検出システムの実行を停止
        """
        self.running = False
