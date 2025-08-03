import cv2

class Camera:
    """
    YOLOによる車体検出のためにカメラデバイスからフレームを取得するクラス
    """
    def __init__(self):
        """
        カメラの初期化
        - self.cap: OpenCVのVideoCaptureオブジェクト
        - self.running: カメラ動作状態を管理するフラグ
        """
        pipeline = self.gstreamer_pipeline()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)  # デフォルトカメラを使用
        self.running = True
    
    def get_frame(self):
        """
        カメラからフレームを取得
        
        Returns:
            ret (bool): フレーム取得成功フラグ
            frame (numpy.ndarray): 取得したフレーム画像
        """
        if not self.running:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def stop(self):
        """
        カメラの動作を停止
        """
        self.running = False
        self.cap.release()

    def gstreamer_pipeline(self):
        return (
            "nvarguscamerasrc sensor_mode=0 ! "
            "video/x-raw(memory:NVMM), width=3840, height=2160, framerate=30/1,format=NV12 ! "
            # これで上下反転
            "nvvidconv flip-method=2 ! "
            "videoconvert ! "
            "video/x-raw ! "
            "appsink"
        )
