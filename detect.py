import os
import time
import glob
import datetime
from model.detector import Detector
import cv2
import config

class PictureDetector:
    """
    保存された画像に対して逐次的に物体検出を行うクラス
    """
    def __init__(self, image_dir=None, enable_fps_monitoring=None, process_interval=None):
        """
        初期化
        
        Args:
            image_dir (str): 画像が保存されているディレクトリ
            enable_fps_monitoring (bool): FPS監視の有効/無効
            process_interval (int): 処理間隔（何枚ごとに処理するか）
        """
        # 設定値をconfig.pyから取得（引数がNoneの場合のみ）
        if image_dir is None:
            image_dir = config.PICTURE_IMAGE_DIR
        if enable_fps_monitoring is None:
            enable_fps_monitoring = config.PICTURE_ENABLE_FPS_MONITORING
        if process_interval is None:
            process_interval = config.PICTURE_PROCESS_INTERVAL

        self.image_dir = image_dir
        self.detector = Detector()
        self.processed_files = set()  # 処理済みファイルを管理
        self.running = False
        
        # 終了条件関連
        self.start_time = datetime.datetime.now()
        self.start_date = self.start_time.date()
        
        # FPS監視関連
        self.enable_fps_monitoring = enable_fps_monitoring
        self.fps_interval = config.PICTURE_FPS_INTERVAL  # FPS計算間隔（枚数）
        self.processing_times = []  # 処理時間の履歴
        self.last_fps_time = time.time()
        self.processed_since_last_fps = 0
        
        # 処理間隔
        self.process_interval = process_interval
        
        # 新しい画像が来た最後の時刻
        self.last_new_file_time = time.time()
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(image_dir, exist_ok=True)
        
        # 日付ディレクトリを作成
        current_date = self.start_date.strftime('%Y-%m-%d')
        self.date_dir = os.path.join(image_dir, current_date)
        os.makedirs(self.date_dir, exist_ok=True)

        # 終了条件用の値を初期化時に計算
        self.exit_time = datetime.datetime.combine(
            self.start_date,
            datetime.time(config.PICTURE_EXIT_HOUR, config.PICTURE_EXIT_MINUTE)
        )
        self.max_runtime_seconds = config.PICTURE_MAX_RUNTIME_HOURS * 3600

        # 画像ファイルのglobパターンを事前に作成
        self.jpg_pattern = os.path.join(self.image_dir, "*.jpg")
    
    def check_exit_conditions(self):
        """
        終了条件をチェック
        
        Returns:
            bool: 終了条件を満たしている場合はTrue
        """
        current_time = datetime.datetime.now()
        current_date = current_time.date()
        
        # 条件1: 実行した日付の指定時刻以降に取得したフォルダの中にファイルがなかった場合
        if current_date == self.start_date:
            if current_time >= self.exit_time:
                # 指定時刻以降の場合、フォルダ内のファイルをチェック
                if os.path.exists(self.image_dir):
                    image_files = glob.glob(self.jpg_pattern)
                    if not image_files:
                        print(f"終了条件1: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {self.exit_time.strftime('%H:%M')}以降でフォルダ内にファイルがありません")
                        return True
        
        # 条件2: 実行してから指定時間後にまだファイルを取得しようとしたとき
        elapsed_time = (current_time - self.start_time).total_seconds()
        if elapsed_time >= self.max_runtime_seconds:
            print(f"終了条件2: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - 実行開始から{config.PICTURE_MAX_RUNTIME_HOURS}時間経過しました")
            return True
        
        return False
    
    def get_new_files(self, max_files=1000):
        """
        新しい画像ファイルを取得
        
        Args:
            max_files (int): 一度に取得する最大ファイル数
            
        Returns:
            list: 新しい画像ファイルのパスのリスト
        """
        if not os.path.exists(self.image_dir):
            return []
        
        # 画像ファイルを取得（UNIX時刻順にソート）
        image_files = glob.glob(self.jpg_pattern)
        image_files.sort()  # UNIX時刻順にソート
        
        # 未処理のファイルのみを取得
        new_files = [f for f in image_files if f not in self.processed_files]
        
        # 新しいファイルがあれば、最終検出時刻を更新
        if new_files:
            self.last_new_file_time = time.time()
        
        # 最大件数に制限
        if len(new_files) > max_files:
            new_files = new_files[:max_files]
            print(f"処理件数を {max_files} 件に制限しました")
        
        return new_files
    
    def process_image(self, image_path):
        """
        単一画像に対して物体検出を実行
        
        Args:
            image_path (str): 画像ファイルのパス
            
        Returns:
            tuple: (is_detected, new_ids, processed_frame)
        """
        try:
            # 画像を読み込み
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"画像の読み込みに失敗: {image_path}")
                return False, [], None
            
            # 物体検出を実行
            is_detected, new_ids, processed_frame = self.detector.detect_object(True, frame)
            
            # ファイル名からUNIX時刻を取得
            filename = os.path.basename(image_path)
            timestamp = filename.replace('.jpg', '')
            
            if is_detected:
                print(f"[{timestamp}] 物体検出: 新しいID {new_ids}")
            else:
                print(f"[{timestamp}] 物体未検出")
            
            return is_detected, new_ids, processed_frame
            
        except Exception as e:
            print(f"画像処理エラー {image_path}: {e}")
            return False, [], None
    
    def delete_processed_files(self):
        """
        処理済みファイルを削除
        """
        files_to_delete = []
        for file_path in self.processed_files:
            if os.path.exists(file_path):
                files_to_delete.append(file_path)
        
        if files_to_delete:
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"ファイル削除エラー {file_path}: {e}")
            
            print(f"処理済みファイル {deleted_count} 件を削除しました")
            
            # 処理済みファイルリストをクリア
            self.processed_files.clear()
    
    def calculate_fps(self, processing_time):
        """
        FPSを計算して表示
        
        Args:
            processing_time (float): 処理時間（秒）
        """
        if not self.enable_fps_monitoring:
            return
        
        # 処理時間を記録
        self.processing_times.append(processing_time)
        self.processed_since_last_fps += 1
        
        # 指定間隔でFPSを計算・表示
        if self.processed_since_last_fps >= self.fps_interval:
            current_time = time.time()
            elapsed_time = current_time - self.last_fps_time
            
            # FPS計算
            fps = self.processed_since_last_fps / elapsed_time
            
            # 平均処理時間計算
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            min_processing_time = min(self.processing_times)
            max_processing_time = max(self.processing_times)
            
            print(f"\n=== FPS統計 ===")
            print(f"処理枚数: {self.processed_since_last_fps} 枚")
            print(f"経過時間: {elapsed_time:.2f} 秒")
            print(f"FPS: {fps:.2f}")
            print(f"平均処理時間: {avg_processing_time:.4f} 秒")
            print(f"最小処理時間: {min_processing_time:.4f} 秒")
            print(f"最大処理時間: {max_processing_time:.4f} 秒")
            print("================")
            
            # 統計をリセット
            self.processing_times.clear()
            self.last_fps_time = current_time
            self.processed_since_last_fps = 0
    
    def save_to_inputs_directory(self, original_frame, original_path):
        """
        検出された車の画像をinputsディレクトリに保存
        
        Args:
            original_frame: 元のフレーム
            original_path (str): 元の画像パス
        """
        if original_frame is None:
            return
        
        try:
            # config.pyからinputsディレクトリを作成
            config.ensure_directories()
            
            # 画像パスから日付を取得
            date_str = config.get_date_from_path(original_path)
            if date_str is None:
                print(f"画像パスから日付を取得できませんでした: {original_path}")
                return
            
            # inputs/日付ディレクトリのパスを取得
            inputs_date_dir = config.get_inputs_date_dir(date_str)
            os.makedirs(inputs_date_dir, exist_ok=True)
            
            # 元のファイル名を取得
            original_filename = os.path.basename(original_path)
            
            # inputsディレクトリに画像を保存
            inputs_save_path = os.path.join(inputs_date_dir, original_filename)
            cv2.imwrite(inputs_save_path, original_frame)
            
            print(f"車検出画像をinputsディレクトリに保存: {inputs_save_path}")
            
        except Exception as e:
            print(f"inputs画像保存エラー: {e}")
    
    def run_detection_loop(self, interval=0.1):
        """
        物体検出ループを実行
        
        Args:
            interval (float): チェック間隔（秒）
        """
        print(f"物体検出開始: 監視ディレクトリ={self.image_dir}")
        print(f"チェック間隔: {interval}秒")
        print("Ctrl+C で停止")
        
        self.running = True
        detection_count = 0
        total_processed = 0
        
        try:
            while self.running:
                # 30分以上新しい画像が来ていない場合は強制的に終了条件を再判定
                if time.time() - self.last_new_file_time >= 1800:
                    print("30分以上新しい画像が来ていません。終了条件を再判定します...")
                    if self.check_exit_conditions():
                        print("終了条件を満たしたため、処理を停止します...")
                        self.running = False
                        break
                    else:
                        # 判定後も新しい画像がなければ、last_new_file_timeを今にリセットして再度30分待つ
                        self.last_new_file_time = time.time()

                # 終了条件をチェック
                if self.check_exit_conditions():
                    print("終了条件を満たしたため、処理を停止します...")
                    self.running = False
                    break

                # 新しいファイルを取得
                new_files = self.get_new_files()
                
                if new_files:
                    print(f"\n新しいファイル {len(new_files)} 件を検出")
                    actual_process_count = len(new_files) // self.process_interval + (1 if len(new_files) % self.process_interval > 0 else 0)
                    print(f"{self.process_interval}枚ごとに処理するため、実際に処理するファイル数: {actual_process_count} 件")
                    
                    # 指定間隔ごとに処理（間引き処理）
                    for i in range(0, len(new_files), self.process_interval):
                        if not self.running:
                            break
                        
                        # 処理する画像を選択
                        image_path = new_files[i]
                        
                        # 処理開始時刻を記録
                        start_time = time.time()
                        
                        # 元の画像を読み込み（inputsディレクトリへの保存用）
                        original_frame = cv2.imread(image_path)
                        
                        # 物体検出を実行
                        is_detected, new_ids, _ = self.process_image(image_path)
                        
                        # 検出された場合のみinputsディレクトリに元画像を保存
                        if is_detected:
                            self.save_to_inputs_directory(original_frame, image_path)

                        # 処理時間を計算
                        processing_time = time.time() - start_time
                        
                        # FPS計算
                        self.calculate_fps(processing_time)
                        
                        # 統計情報を更新
                        total_processed += 1
                        if is_detected:
                            detection_count += 1
                        
                        # ファイルを処理済みとしてマーク（指定間隔分）
                        for j in range(i, min(i + self.process_interval, len(new_files))):
                            self.processed_files.add(new_files[j])
                        
                        # 少し待機（連続処理を避ける）
                        time.sleep(0.01)
                    
                    # 統計情報を表示
                    if total_processed > 0:
                        detection_rate = (detection_count / total_processed) * 100
                        print(f"統計: 処理済み {total_processed} 件, 検出 {detection_count} 件, 検出率 {detection_rate:.1f}%")
                    
                    # 処理済みファイルを削除
                    self.delete_processed_files()
                
                # 指定間隔で待機
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n物体検出を停止します...")
        finally:
            self.stop()
    
    def stop(self):
        """
        物体検出を停止
        """
        self.running = False
        self.detector.stop()
        print(f"処理完了: 総処理ファイル数 {len(self.processed_files)}")

def main():
    """
    メイン関数
    """
    # 設定
    enable_fps_monitoring = True
    process_interval = 3  # 3枚ごとに処理
    
    # 物体検出器の作成
    detector = PictureDetector(
        enable_fps_monitoring=enable_fps_monitoring,
        process_interval=process_interval
    )
    
    print("保存された画像に対する物体検出テスト")
    print("test_save_rate.py で画像を保存しながら、このプログラムで検出を実行してください")
    
    if enable_fps_monitoring:
        print(f"FPS監視: 有効 (100枚ごとに統計表示)")
    else:
        print("FPS監視: 無効")
    
    print(f"処理間隔: {process_interval}枚ごと")
    
    # 物体検出ループを開始
    detector.run_detection_loop()

if __name__ == "__main__":
    main()
