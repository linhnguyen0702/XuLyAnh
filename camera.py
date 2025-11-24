# Import OpenCV để xử lý video và ảnh
import cv2
# Import class model nhận diện cảm xúc
from model import FacialExpressionModel
# Import numpy để xử lý mảng
import numpy as np
# Import DebugVisualizer và TerminalLogger từ image_processor
from image_processor import DebugVisualizer, TerminalLogger

# --- Khởi tạo một lần duy nhất (global) để tối ưu hiệu năng ---
# Load Haar Cascade Classifier để phát hiện khuôn mặt
# Khởi tạo ở global scope để chỉ load 1 lần, không phải load lại mỗi frame
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load model CNN để nhận diện cảm xúc
# Khởi tạo ở global scope để chỉ load 1 lần, tránh tốn thời gian load lại
model = FacialExpressionModel("model.json", "model_weights.h5")
# Chọn font để vẽ text
font = cv2.FONT_HERSHEY_SIMPLEX
# ----------------------------------

class VideoCamera(object):
    def __init__(self, show_debug_window=True, enable_terminal_log=True):
        """
        Args:
            show_debug_window: Bật/tắt hiển thị cửa sổ debug
            enable_terminal_log: Bật/tắt in log ra terminal
        """
        # Mở camera với index 0 (camera mặc định)
        # VideoCapture(0) = camera đầu tiên trong hệ thống
        self.video = cv2.VideoCapture(0)
        # Kiểm tra xem camera có mở được không
        if not self.video.isOpened():
            # Nếu không mở được, raise exception
            raise RuntimeError("Could not start camera.")
        
        # Khởi tạo debug visualizer
        self.debug_visualizer = DebugVisualizer(enabled=show_debug_window)
        
        # Khởi tạo terminal logger
        self.logger = TerminalLogger(enabled=enable_terminal_log)
        
        # Biến để theo dõi frame đầu tiên
        self.first_frame = True
        
        # *** THÊM: Lưu trữ dữ liệu để hiển thị debug khi dừng ***
        # Lưu trữ các frame đã xử lý và dữ liệu tương ứng
        self.collected_frames = []  # Danh sách [(original_frame, gray_frame, faces, rois, predictions, final_frame), ...]
        self.show_debug_on_stop = show_debug_window  # Cờ để biết có hiển thị debug khi dừng không

    def __del__(self):
        """
        Destructor: Tự động giải phóng camera khi đối tượng bị hủy
        Đảm bảo camera luôn được giải phóng đúng cách
        """
        # Kiểm tra camera còn mở không
        if self.video.isOpened():
            # Giải phóng camera
            self.video.release()
    
    def reset_collected_frames(self):
        """
        Reset danh sách các frame đã thu thập
        Gọi method này khi bắt đầu session webcam mới
        """
        self.collected_frames = []
        self.first_frame = True
        self.logger.info("Đã reset danh sách frame thu thập")

    def release(self):
        """
        Phương thức giải phóng camera một cách tường minh
        Có thể gọi thủ công khi cần dừng camera
        """
        # *** THÊM: Hiển thị tất cả debug windows khi dừng webcam ***
        if self.show_debug_on_stop and len(self.collected_frames) > 0:
            self.logger.separator()
            self.logger.info(f"Đã thu thập {len(self.collected_frames)} frame")
            
            # *** NHÓM CÁC FRAME THEO CẢM XÚC ***
            # Dictionary để lưu frame đại diện cho mỗi cảm xúc
            # Key: tên cảm xúc, Value: (frame_data, count)
            emotion_groups = {}
            
            # Đếm số lượng mỗi cảm xúc và lấy frame đầu tiên làm đại diện
            for frame_data in self.collected_frames:
                original_frame, gray_frame, faces, rois, predictions, final_frame = frame_data
                
                # Duyệt qua các cảm xúc trong frame này
                for pred in predictions:
                    if pred not in emotion_groups:
                        # Lưu frame đầu tiên của cảm xúc này làm đại diện
                        # Lọc chỉ lấy ROI và prediction của cảm xúc này
                        filtered_rois = [(roi, emotion) for roi, emotion in rois if emotion == pred]
                        emotion_groups[pred] = {
                            'frame_data': (original_frame, gray_frame, faces, filtered_rois, [pred], final_frame),
                            'count': 1
                        }
                    else:
                        # Tăng số đếm cho cảm xúc này
                        emotion_groups[pred]['count'] += 1
            
            self.logger.info(f"Phát hiện {len(emotion_groups)} loại cảm xúc khác nhau")
            self.logger.info("*** NHẤN PHÍM BẤT KỲ trong cửa sổ để chuyển sang cảm xúc tiếp theo ***")
            self.logger.separator()
            
            # Hiển thị 1 cửa sổ cho mỗi loại cảm xúc
            for idx, (emotion, data) in enumerate(emotion_groups.items(), 1):
                frame_data = data['frame_data']
                count = data['count']
                
                self.logger.info(f"Cửa sổ {idx}/{len(emotion_groups)}: {emotion} (phát hiện {count} lần)")
                
                original_frame, gray_frame, faces, filtered_rois, predictions, final_frame = frame_data
                
                # Hiển thị debug window cho cảm xúc này
                self.debug_visualizer.show_processing_steps(
                    original_img=original_frame,
                    gray_img=gray_frame,
                    faces=faces,
                    rois=filtered_rois,
                    predictions=predictions,
                    final_img=final_frame
                )
            
            self.logger.success(f"Đã hiển thị {len(emotion_groups)} cửa sổ debug (1 cửa sổ/cảm xúc)")
            
            # In thống kê chi tiết
            self.logger.separator()
            self.logger.info("THỐNG KÊ CẢM XÚC:")
            total_detections = sum(data['count'] for data in emotion_groups.values())
            for emotion, data in sorted(emotion_groups.items(), key=lambda x: x[1]['count'], reverse=True):
                count = data['count']
                percentage = (count / total_detections) * 100
                self.logger.info(f"  • {emotion}: {count} lần ({percentage:.1f}%)")
            self.logger.separator()
            
            # Reset danh sách
            self.collected_frames = []
        
        # Đóng debug window
        self.debug_visualizer.close()
        # Kiểm tra camera còn mở không
        if self.video.isOpened():
            # Giải phóng camera
            self.video.release()
            print("Camera released.")

    def get_frame(self):
        """
        Đọc một frame từ webcam, xử lý và trả về frame đã encode
        Returns:
            (jpeg_bytes, emotions_detected): Tuple chứa frame JPEG và danh sách cảm xúc
        """
        # Đọc một frame từ webcam
        # success: True nếu đọc được, False nếu không
        # frame: ảnh frame ở định dạng BGR
        success, frame = self.video.read()
        
        # Nếu không đọc được frame (camera lỗi hoặc đã đóng)
        if not success:
            # Tạo frame trống màu đen (480x640 pixels, 3 channels BGR)
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Encode frame trống thành JPEG
            _, jpeg = cv2.imencode('.jpg', blank_frame)
            # Trả về frame trống và danh sách cảm xúc rỗng
            return jpeg.tobytes(), []

        # Log frame đầu tiên
        if self.first_frame:
            self.logger.start_processing("WEBCAM REALTIME", "Camera index 0")
            self.first_frame = False

        # Tạo bản copy của frame để vẽ lên, đảm bảo frame gốc không bị thay đổi
        # (phòng trường hợp frame gốc cần dùng cho mục đích khác)
        processed_frame = frame.copy()
        # Lưu frame gốc để hiển thị debug
        original_frame = frame.copy()
        
        # Bước 1: Chuyển frame từ BGR sang grayscale
        # Haar Cascade hoạt động tốt hơn trên ảnh grayscale
        gray_fr = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        # Bước 2: Phát hiện khuôn mặt trong frame
        # scaleFactor=1.3: Tỷ lệ giảm kích thước mỗi lần quét (1.3 = giảm 30%)
        # minNeighbors=5: Số lượng lân cận tối thiểu để xác nhận là khuôn mặt
        # Trả về danh sách các hình chữ nhật (x, y, width, height)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        # Khởi tạo danh sách lưu các cảm xúc phát hiện được
        emotions_detected = []
        # Lưu danh sách ROI và predictions để hiển thị debug
        rois_list = []
        predictions_list = []
        
        # Log thông tin frame (chỉ log khi có khuôn mặt để không spam)
        if len(faces) > 0:
            self.logger.info(f"Frame: Phát hiện {len(faces)} khuôn mặt")
        
        # Bước 3: Duyệt qua từng khuôn mặt đã phát hiện
        for idx, (x, y, w, h) in enumerate(faces, 1):
            if len(faces) > 0:
                self.logger.info(f"  Khuôn mặt #{idx}: Vị trí ({x}, {y}), Kích thước {w}x{h}")
            
            # Bước 3.1: Cắt vùng khuôn mặt (ROI) từ ảnh grayscale
            # gray_fr[y:y+h, x:x+w] = cắt từ dòng y đến y+h, cột x đến x+w
            fc = gray_fr[y:y+h, x:x+w]
            
            # Bước 3.2: Resize vùng khuôn mặt về 48x48 pixels
            # Model CNN yêu cầu input size 48x48
            roi = cv2.resize(fc, (48, 48))
            
            # Bước 3.3: Chuẩn hóa shape và dự đoán cảm xúc
            # roi shape: (48, 48) → (1, 48, 48, 1) [batch, height, width, channel]
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            
            if len(faces) > 0:
                self.logger.success(f"    └─ Cảm xúc: {pred}")
            
            # Lưu ROI và prediction để hiển thị debug
            rois_list.append((roi.copy(), pred))
            predictions_list.append(pred)
            
            # Thêm cảm xúc vào danh sách
            emotions_detected.append(pred)

            # Bước 3.4: Vẽ text cảm xúc lên frame
            # (x, y-10): vị trí text (10 pixels phía trên khung)
            # font: font chữ, 0.8: kích thước font
            # (255, 255, 0): màu BGR (vàng), 2: độ dày chữ
            cv2.putText(processed_frame, pred, (x, y - 10), font, 0.8, (255, 255, 0), 2)
            
            # Bước 3.5: Vẽ hình chữ nhật bao quanh khuôn mặt
            # (x, y): điểm góc trên bên trái, (x+w, y+h): điểm góc dưới bên phải
            # (255, 0, 0): màu BGR (xanh dương), 2: độ dày đường viền
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # *** SỬA: Lưu trữ frame thay vì hiển thị ngay ***
        # Chỉ lưu frame khi có khuôn mặt được phát hiện
        if len(faces) > 0 and len(predictions_list) > 0:
            # Lưu tất cả thông tin cần thiết để hiển thị debug sau này
            frame_data = (
                original_frame.copy(),  # Ảnh gốc
                gray_fr.copy(),         # Ảnh grayscale
                faces.copy(),           # Danh sách khuôn mặt
                rois_list.copy(),       # Danh sách ROI
                predictions_list.copy(), # Danh sách cảm xúc
                processed_frame.copy()  # Ảnh cuối cùng
            )
            self.collected_frames.append(frame_data)
            self.logger.info(f"Đã lưu frame #{len(self.collected_frames)} để hiển thị sau khi dừng webcam")

        # Bước 4: Encode frame đã xử lý thành định dạng JPEG
        # JPEG là format phổ biến để truyền qua mạng (kích thước nhỏ)
        _, jpeg = cv2.imencode('.jpg', processed_frame)
        
        # Trả về frame đã encode (bytes) và danh sách cảm xúc
        return jpeg.tobytes(), emotions_detected


if __name__ == "__main__":
    # Chế độ chạy trực tiếp để debug: hiển thị cửa sổ phân tích thời gian thực
    # Chạy: python camera.py
    cam = None
    try:
        cam = VideoCamera(show_debug_window=True, enable_terminal_log=True)
        cam.reset_collected_frames()

        print("Bắt đầu webcam. Nhấn 'q' để dừng và hiển thị cửa sổ debug.")
        while True:
            jpeg_bytes, emotions = cam.get_frame()

            # Decode JPEG bytes thành ảnh BGR để hiển thị bằng OpenCV
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Hiển thị frame
            cv2.imshow('Emotion Analysis', frame)

            # Nhấn 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Lỗi khi mở camera: {e}")
    finally:
        # Giải phóng camera và đóng cửa sổ
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
