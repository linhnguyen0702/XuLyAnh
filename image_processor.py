# Import thư viện OpenCV để xử lý ảnh và video
import cv2
# Import numpy để xử lý mảng đa chiều
import numpy as np
# Import class model nhận diện cảm xúc
from model import FacialExpressionModel
# Import imageio để đọc ghi video mạnh mẽ hơn (hỗ trợ nhiều codec)
import imageio
# Import time để đồng bộ FPS khi xử lý video
import time
# Import datetime để hiển thị thời gian
from datetime import datetime

class TerminalLogger:
    """
    Class để in log chi tiết quá trình xử lý ảnh ra terminal/console
    """
    # ANSI color codes cho terminal
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    def __init__(self, enabled=True):
        """
        Args:
            enabled: Bật/tắt logging
        """
        self.enabled = enabled
    
    def _get_timestamp(self):
        """Lấy timestamp hiện tại"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print(self, message, color=WHITE, bold=False):
        """In message với màu"""
        if not self.enabled:
            return
        prefix = self.BOLD if bold else ""
        print(f"{prefix}{color}[{self._get_timestamp()}] {message}{self.RESET}")
    
    def step(self, step_num, description, details=""):
        """In một bước xử lý"""
        self._print(f"BƯỚC {step_num}: {description}", self.CYAN, bold=True)
        if details:
            self._print(f"  └─ {details}", self.WHITE)
    
    def info(self, message):
        """In thông tin"""
        self._print(f"ℹ {message}", self.BLUE)
    
    def success(self, message):
        """In thành công"""
        self._print(f"✓ {message}", self.GREEN)
    
    def warning(self, message):
        """In cảnh báo"""
        self._print(f"⚠ {message}", self.YELLOW)
    
    def error(self, message):
        """In lỗi"""
        self._print(f"✗ {message}", self.RED, bold=True)
    
    def separator(self):
        """In đường phân cách"""
        if self.enabled:
            print(f"{self.CYAN}{'='*60}{self.RESET}")
    
    def start_processing(self, processing_type, source=""):
        """Bắt đầu quá trình xử lý"""
        self.separator()
        self._print(f"🚀 BẮT ĐẦU XỬ LÝ: {processing_type}", self.MAGENTA, bold=True)
        if source:
            self._print(f"📁 Nguồn: {source}", self.WHITE)
        self.separator()
    
    def end_processing(self, processing_type, emotion_counts=None):
        """Kết thúc quá trình xử lý"""
        self.separator()
        self._print(f"✅ HOÀN THÀNH XỬ LÝ: {processing_type}", self.GREEN, bold=True)
        if emotion_counts:
            total = sum(emotion_counts.values())
            if total > 0:
                self._print(f"📊 Tổng số khuôn mặt phát hiện: {total}", self.CYAN)
                for emotion, count in emotion_counts.items():
                    if count > 0:
                        percentage = (count / total) * 100
                        self._print(f"  • {emotion}: {count} ({percentage:.1f}%)", self.WHITE)
        self.separator()
        print()  # Dòng trống

class DebugVisualizer:
    """
    Class để hiển thị từng bước xử lý ảnh trong cửa sổ OpenCV riêng
    Giúp debug và hiểu rõ quá trình xử lý
    """
    def __init__(self, enabled=True):
        """
        Args:
            enabled: Bật/tắt hiển thị debug window
        """
        self.enabled = enabled
        # Sử dụng tiếng Việt không dấu
        self.window_name = "Qua trinh xu ly anh - Debug Window"
        self.window_created = False  # Theo dõi xem cửa sổ đã được tạo chưa
    
    def show_processing_steps(self, original_img, gray_img, faces, rois, predictions, final_img):
        """
        Hiển thị tất cả các bước xử lý trong một cửa sổ
        Chỉ hiển thị khi đã nhận diện thành công (có kết quả)
        Args:
            original_img: Ảnh gốc (BGR)
            gray_img: Ảnh grayscale
            faces: Danh sách khuôn mặt phát hiện được [(x, y, w, h), ...]
            rois: Danh sách ROI đã resize [(roi_48x48, emotion), ...]
            predictions: Danh sách cảm xúc dự đoán
            final_img: Ảnh cuối cùng đã vẽ kết quả
        """
        if not self.enabled:
            return
        
        # Chỉ hiển thị khi đã có kết quả nhận diện (có faces hoặc đã xử lý xong)
        # Không hiển thị nếu chưa có gì để hiển thị
        if len(faces) == 0 and len(predictions) == 0:
            print(f"[DEBUG] Không hiển thị debug window: faces={len(faces)}, predictions={len(predictions)}")
            return
        
        print(f"[DEBUG] Đang hiển thị debug window: faces={len(faces)}, predictions={len(predictions)}")
        
        # Chỉ tạo cửa sổ khi cần hiển thị lần đầu tiên
        if not self.window_created:
            try:
                print(f"[DEBUG] Đang tạo cửa sổ debug: {self.window_name}")
                # Tạo cửa sổ debug với kích thước lớn hơn để hiển thị rõ ràng
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1400, 900)
                # Đặt vị trí cửa sổ (tùy chọn)
                cv2.moveWindow(self.window_name, 100, 100)
                self.window_created = True
                print(f"[DEBUG] Đã tạo cửa sổ debug thành công")
            except Exception as e:
                print(f"[ERROR] Không thể tạo cửa sổ debug: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Tạo ảnh để hiển thị các bước với kích thước lớn hơn
        h, w = original_img.shape[:2]
        display_h = 350
        display_w = 450
        
        # Resize các ảnh để hiển thị
        original_display = cv2.resize(original_img, (display_w, display_h))
        gray_display = cv2.cvtColor(cv2.resize(gray_img, (display_w, display_h)), cv2.COLOR_GRAY2BGR)
        
        # Tạo ảnh với khung phát hiện khuôn mặt (màu xanh lá)
        detection_img = original_img.copy()
        for (x, y, w_face, h_face) in faces:
            cv2.rectangle(detection_img, (x, y), (x+w_face, y+h_face), (0, 255, 0), 3)
            # Vẽ text "Face" trên khung
            cv2.putText(detection_img, "Face", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        detection_display = cv2.resize(detection_img, (display_w, display_h))
        
        # Tạo ảnh cuối cùng
        final_display = cv2.resize(final_img, (display_w, display_h))
        
        # Tạo grid để hiển thị các bước
        # Hàng 1: Ảnh gốc, Grayscale, Phát hiện khuôn mặt
        row1 = np.hstack([original_display, gray_display, detection_display])
        
        # Hàng 2: ROI và kết quả
        # Tạo ảnh hiển thị các ROI với kích thước lớn hơn
        roi_display_list = []
        roi_size = 150  # Kích thước ROI lớn hơn để dễ nhìn
        for i, (roi, emotion) in enumerate(rois[:3]):  # Hiển thị tối đa 3 ROI
            roi_colored = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            # Phóng to ROI để dễ nhìn (48x48 -> 150x150)
            roi_resized = cv2.resize(roi_colored, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)
            # Vẽ khung xung quanh ROI
            cv2.rectangle(roi_resized, (0, 0), (roi_size-1, roi_size-1), (0, 255, 255), 2)
            # Vẽ text cảm xúc với font lớn hơn
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (roi_size - text_size[0]) // 2
            cv2.putText(roi_resized, emotion, (text_x, roi_size - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            roi_display_list.append(roi_resized)
        
        # Nếu không đủ 3 ROI, thêm ảnh trống
        while len(roi_display_list) < 3:
            blank = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            cv2.putText(blank, "No face", (20, roi_size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            roi_display_list.append(blank)
        
        # Ghép các ROI thành một hàng với khoảng cách
        roi_spacing = 20
        roi_row = np.hstack([roi_display_list[0], 
                            np.zeros((roi_size, roi_spacing, 3), dtype=np.uint8),
                            roi_display_list[1],
                            np.zeros((roi_size, roi_spacing, 3), dtype=np.uint8),
                            roi_display_list[2]])
        # Resize để khớp với display_w
        roi_row_height = roi_size + 40
        roi_row = cv2.resize(roi_row, (display_w, roi_row_height))
        
        # Tạo ảnh thông tin với nền đẹp hơn
        info_img = np.ones((display_h, display_w, 3), dtype=np.uint8) * 240  # Nền xám nhạt
        # Vẽ khung viền
        cv2.rectangle(info_img, (0, 0), (display_w-1, display_h-1), (100, 100, 100), 2)
        
        y_offset = 40
        # Tiêu đề
        cv2.putText(info_img, "THONG TIN XU LY", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y_offset += 40
        
        # Thông tin số khuôn mặt
        cv2.putText(info_img, f"Khuon mat phat hien: {len(faces)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        y_offset += 40
        
        # Danh sách cảm xúc
        cv2.putText(info_img, "Cam xuc du doan:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 35
        
        if len(predictions) > 0:
            for i, emotion in enumerate(predictions[:6]):  # Hiển thị tối đa 6 cảm xúc
                # Màu sắc cho từng cảm xúc
                colors = {
                    "Angry": (0, 0, 255),      # Đỏ
                    "Disgust": (128, 0, 128),  # Tím
                    "Fear": (0, 165, 255),     # Cam
                    "Happy": (0, 255, 0),      # Xanh lá
                    "Neutral": (128, 128, 128), # Xám
                    "Sad": (255, 0, 0),        # Xanh dương
                    "Surprise": (255, 192, 203) # Hồng
                }
                color = colors.get(emotion, (0, 0, 0))
                cv2.putText(info_img, f"{i+1}. {emotion}", (30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 30
        else:
            cv2.putText(info_img, "Khong co khuon mat", (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Hàng 2: ROI và thông tin
        # Đảm bảo roi_row và final_display có cùng chiều rộng trước khi ghép
        # Resize final_display để có cùng chiều rộng với roi_row
        final_display_resized = cv2.resize(final_display, (display_w, display_h))
        
        # Ghép roi_row và final_display (cả hai đều có chiều rộng display_w)
        # Đảm bảo cả hai có cùng chiều rộng
        if roi_row.shape[1] != final_display_resized.shape[1]:
            # Nếu chiều rộng không khớp, resize lại
            roi_row = cv2.resize(roi_row, (final_display_resized.shape[1], roi_row.shape[0]))
        
        row2_left = np.vstack([roi_row, final_display_resized])
        
        # Đảm bảo row2_left và info_img có cùng chiều cao trước khi ghép ngang
        if row2_left.shape[0] != info_img.shape[0]:
            # Nếu chiều cao không khớp, resize info_img
            info_img = cv2.resize(info_img, (info_img.shape[1], row2_left.shape[0]))
        
        # Ghép với info_img (cả hai đều có cùng chiều cao)
        row2 = np.hstack([row2_left, info_img])
        
        # Đảm bảo row1 và row2 có cùng chiều rộng trước khi ghép dọc
        if row1.shape[1] != row2.shape[1]:
            # Nếu chiều rộng không khớp, resize row2
            row2 = cv2.resize(row2, (row1.shape[1], row2.shape[0]))
        
        # Ghép tất cả lại
        display = np.vstack([row1, row2])
        
        # Vẽ tiêu đề cho từng bước với nền đen để dễ đọc
        title_bg_h = 40
        # Vẽ nền đen cho tiêu đề
        cv2.rectangle(display, (0, 0), (display_w, title_bg_h), (0, 0, 0), -1)
        cv2.rectangle(display, (display_w, 0), (display_w*2, title_bg_h), (0, 0, 0), -1)
        cv2.rectangle(display, (display_w*2, 0), (display_w*3, title_bg_h), (0, 0, 0), -1)
        cv2.rectangle(display, (0, display_h), (display_w, display_h + title_bg_h), (0, 0, 0), -1)
        cv2.rectangle(display, (0, display_h + roi_size + 40), (display_w, display_h + roi_size + 40 + title_bg_h), (0, 0, 0), -1)
        
        # Vẽ text tiêu đề
        cv2.putText(display, "1. ANH GOC", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "2. GRAYSCALE", (display_w + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "3. PHAT HIEN KHUON MAT", (display_w*2 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "4. ROI (48x48)", (15, display_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "5. KET QUA CUOI CUNG", (15, display_h + roi_size + 40 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị cửa sổ debug
        try:
            print(f"[DEBUG] Đang hiển thị ảnh trong cửa sổ debug...")
            print(f"[DEBUG] Kích thước display: {display.shape}")
            cv2.imshow(self.window_name, display)
            
            # *** QUAN TRỌNG: Sử dụng waitKey(0) để giữ cửa sổ mở ***
            # waitKey(0) = Đợi vô hạn cho đến khi người dùng nhấn phím
            # Cửa sổ sẽ hiển thị và đợi người dùng xem xong
            print(f"[DEBUG] Đã hiển thị cửa sổ debug thành công")
            print(f"[DEBUG] Cửa sổ '{self.window_name}' đã được hiển thị.")
            print(f"[DEBUG] *** NHẤN PHÍM BẤT KỲ trong cửa sổ để tiếp tục ***")
            
            # Đợi người dùng nhấn phím để đóng cửa sổ
            cv2.waitKey(0)
            
            # Sau khi người dùng nhấn phím, đóng cửa sổ
            print(f"[DEBUG] Người dùng đã nhấn phím, đóng cửa sổ debug...")
        except Exception as e:
            print(f"[ERROR] Lỗi khi hiển thị cửa sổ debug: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Đóng cửa sổ debug"""
        if self.enabled and self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
                self.window_created = False
            except:
                pass

class ImageProcessor:
    def __init__(self, show_debug_window=True, enable_terminal_log=True):
        """
        Args:
            show_debug_window: Bật/tắt hiển thị cửa sổ debug
            enable_terminal_log: Bật/tắt in log ra terminal
        """
        # Khởi tạo Haar Cascade Classifier để phát hiện khuôn mặt
        # File XML chứa các đặc trưng (features) đã được train sẵn
        self.facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Khởi tạo model CNN để nhận diện cảm xúc
        # model.json: kiến trúc mạng neural network
        # model_weights.h5: trọng số đã được train
        self.model = FacialExpressionModel("model.json", "model_weights.h5")
        
        # Chọn font để vẽ text cảm xúc lên ảnh
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Khởi tạo debug visualizer
        self.debug_visualizer = DebugVisualizer(enabled=show_debug_window)
        
        # Khởi tạo terminal logger
        self.logger = TerminalLogger(enabled=enable_terminal_log)

    def process_image(self, image_path, output_path):
        """
        Xử lý một ảnh tĩnh và nhận diện cảm xúc
        Args:
            image_path: Đường dẫn đến ảnh đầu vào
            output_path: Đường dẫn để lưu ảnh đã xử lý
        Returns:
            output_path: Đường dẫn ảnh đã xử lý
            emotion_counts: Dictionary đếm số lượng từng cảm xúc
        """
        # Bắt đầu logging
        self.logger.start_processing("XỬ LÝ ẢNH TĨNH", image_path)
        
        # Bước 1: Đọc ảnh từ đường dẫn bằng OpenCV
        # cv2.imread() trả về ảnh ở định dạng BGR (Blue-Green-Red)
        self.logger.step(1, "Đọc ảnh từ file", f"Đường dẫn: {image_path}")
        img = cv2.imread(image_path)
        
        # Kiểm tra nếu OpenCV không đọc được ảnh (có thể do format không hỗ trợ)
        if img is None:
            self.logger.warning("OpenCV không đọc được ảnh, thử dùng PIL...")
            try:
                # Thử đọc ảnh bằng thư viện PIL (Pillow) - hỗ trợ nhiều format hơn
                from PIL import Image
                # Mở ảnh bằng PIL
                pil_image = Image.open(image_path)
                # Kiểm tra mode của ảnh, nếu không phải RGB thì chuyển sang RGB
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                # Chuyển PIL Image sang numpy array, sau đó chuyển từ RGB sang BGR cho OpenCV
                # OpenCV dùng BGR, PIL dùng RGB nên cần chuyển đổi
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.logger.success(f"Đọc ảnh thành công bằng PIL")
                self.logger.info(f"Kích thước ảnh: {img.shape[1]}x{img.shape[0]} pixels")
            except Exception as e:
                # Nếu cả PIL cũng không đọc được, trả về lỗi
                self.logger.error(f"Không thể đọc ảnh: {str(e)}")
                return None, {"error": f"Cannot read image: {str(e)}"}
        else:
            self.logger.success(f"Đọc ảnh thành công bằng OpenCV")
            self.logger.info(f"Kích thước ảnh: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Bước 2: Chuyển ảnh từ BGR sang grayscale (ảnh xám)
        # Haar Cascade hoạt động tốt hơn trên ảnh grayscale
        self.logger.step(2, "Chuyển đổi sang Grayscale", "BGR → Grayscale")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.logger.success("Chuyển đổi grayscale hoàn tất")
        
        # Bước 3: Phát hiện khuôn mặt bằng Haar Cascade
        # detectMultiScale() trả về danh sách các hình chữ nhật (x, y, width, height)
        # scaleFactor=1.1: Tỷ lệ giảm kích thước ảnh mỗi lần quét (1.1 = giảm 10%)
        # minNeighbors=3: Số lượng lân cận tối thiểu để xác nhận là khuôn mặt
        # minSize=(30, 30): Kích thước khuôn mặt nhỏ nhất có thể phát hiện
        self.logger.step(3, "Phát hiện khuôn mặt", "Sử dụng Haar Cascade Classifier")
        faces = self.facec.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        # Nếu không tìm thấy khuôn mặt với tham số chuẩn, thử lại với tham số nhạy hơn
        if len(faces) == 0:
            self.logger.warning("Không tìm thấy khuôn mặt với tham số chuẩn, thử lại với tham số nhạy hơn...")
            # scaleFactor nhỏ hơn (1.05) = quét kỹ hơn, minSize nhỏ hơn (20,20) = phát hiện khuôn mặt nhỏ hơn
            faces = self.facec.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        
        if len(faces) > 0:
            self.logger.success(f"Phát hiện được {len(faces)} khuôn mặt")
        else:
            self.logger.warning("Không phát hiện được khuôn mặt nào trong ảnh")
        
        # Khởi tạo dictionary để đếm số lượng từng loại cảm xúc
        emotion_counts = {
            "Angry": 0,      # Tức giận
            "Disgust": 0,    # Ghê tởm
            "Fear": 0,       # Sợ hãi
            "Happy": 0,      # Vui vẻ
            "Neutral": 0,    # Trung tính
            "Sad": 0,        # Buồn bã
            "Surprise": 0    # Ngạc nhiên
        }
        
        # Lưu danh sách ROI và predictions để hiển thị debug
        rois_list = []
        predictions_list = []
        
        # Bước 4: Duyệt qua từng khuôn mặt đã phát hiện được
        # (x, y): Tọa độ góc trên bên trái của hình chữ nhật
        # (w, h): Chiều rộng và chiều cao của hình chữ nhật
        self.logger.step(4, "Xử lý từng khuôn mặt và nhận diện cảm xúc", f"Tổng cộng {len(faces)} khuôn mặt")
        for idx, (x, y, w, h) in enumerate(faces, 1):
            self.logger.info(f"  Khuôn mặt #{idx}: Vị trí ({x}, {y}), Kích thước {w}x{h} pixels")
            
            # Bước 4.1: Cắt vùng khuôn mặt (Region of Interest - ROI) từ ảnh grayscale
            # gray_img[y:y+h, x:x+w] = cắt từ dòng y đến y+h, cột x đến x+w
            fc = gray_img[y:y+h, x:x+w]
            self.logger.info(f"    └─ Đã cắt ROI: {w}x{h} pixels")
            
            # Bước 4.2: Resize vùng khuôn mặt về kích thước 48x48 pixels
            # Model CNN được train với input size 48x48, nên cần resize về đúng kích thước này
            roi = cv2.resize(fc, (48, 48))
            self.logger.info(f"    └─ Đã resize ROI về 48x48 pixels")
            
            # Bước 4.3: Chuẩn hóa shape và dự đoán cảm xúc
            # roi shape hiện tại: (48, 48)
            # np.newaxis thêm chiều mới: (1, 48, 48, 1)
            # - Chiều đầu: batch size (1 = 1 ảnh)
            # - Chiều 2,3: height, width (48, 48)
            # - Chiều cuối: channels (1 = grayscale)
            self.logger.info(f"    └─ Đang dự đoán cảm xúc bằng CNN model...")
            pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            self.logger.success(f"    └─ Kết quả: {pred}")
            
            # Lưu ROI và prediction để hiển thị debug
            rois_list.append((roi.copy(), pred))
            predictions_list.append(pred)
            
            # Bước 4.4: Tăng số đếm cho cảm xúc được dự đoán
            emotion_counts[pred] += 1
            
            # Bước 4.5: Vẽ hình chữ nhật bao quanh khuôn mặt
            # (x, y): điểm góc trên bên trái, (x+w, y+h): điểm góc dưới bên phải
            # (255, 0, 0): màu BGR (xanh dương), 2: độ dày đường viền
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Bước 4.6: Vẽ text hiển thị cảm xúc lên ảnh
            # (x, y-10): vị trí text (10 pixels phía trên khung)
            # self.font: font chữ, 0.9: kích thước font
            # (255, 255, 0): màu BGR (vàng), 2: độ dày chữ
            cv2.putText(img, pred, (x, y-10), self.font, 0.9, (255, 255, 0), 2)
        
        # Chỉ hiển thị debug window khi đã nhận diện thành công (có kết quả)
        # Bọc trong try-except để không làm gián đoạn quá trình xử lý nếu có lỗi
        print(f"[DEBUG] Kiểm tra điều kiện hiển thị debug: faces={len(faces)}, predictions={len(predictions_list)}")
        if len(faces) > 0 or len(predictions_list) > 0:
            print(f"[DEBUG] Điều kiện đúng, đang gọi show_processing_steps...")
            try:
                self.debug_visualizer.show_processing_steps(
                    original_img=img.copy(),
                    gray_img=gray_img,
                    faces=faces,
                    rois=rois_list,
                    predictions=predictions_list,
                    final_img=img
                )
                print(f"[DEBUG] Đã gọi show_processing_steps thành công")
            except Exception as e:
                # Nếu có lỗi khi hiển thị debug window, chỉ in ra log, không làm gián đoạn
                self.logger.warning(f"Lỗi khi hiển thị debug window: {str(e)}")
                print(f"[ERROR] Không thể hiển thị debug window: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] Không hiển thị debug window: không có faces hoặc predictions")
        
        # Bước 5: Lưu ảnh đã xử lý (có vẽ khung và text) vào đường dẫn output
        self.logger.step(5, "Lưu ảnh đã xử lý", f"Đường dẫn: {output_path}")
        try:
            cv2.imwrite(output_path, img)
            self.logger.success("Đã lưu ảnh thành công")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu ảnh: {str(e)}")
            # Vẫn trả về kết quả ngay cả khi lưu ảnh lỗi
        
        # Kết thúc logging
        self.logger.end_processing("XỬ LÝ ẢNH TĨNH", emotion_counts)
        
        # Trả về đường dẫn ảnh đã xử lý và dictionary đếm cảm xúc
        # Đảm bảo luôn trả về kết quả, kể cả khi có lỗi
        return output_path, emotion_counts

    def process_video_stream(self, video_path):
        """
        Xử lý video và stream từng frame đã xử lý về browser, đồng bộ với FPS gốc của video
        Args:
            video_path: Đường dẫn đến file video
        Yields:
            (frame_bytes, frame_emotions): Tuple chứa frame đã encode và danh sách cảm xúc
        """
        # *** THÊM: Lưu trữ dữ liệu để hiển thị debug sau khi xử lý xong ***
        collected_frames = []  # Danh sách lưu frame để hiển thị debug
        
        # Bắt đầu logging
        self.logger.start_processing("XỬ LÝ VIDEO STREAMING", video_path)
        
        try:
            # Mở video bằng imageio (hỗ trợ nhiều codec hơn OpenCV)
            reader = imageio.get_reader(video_path)
        except Exception as e:
            # Nếu không mở được video, in lỗi và return
            self.logger.error(f"Không thể mở video: {e}")
            print(f"ERROR: Cannot open video with imageio: {e}")
            return

        try:
            # Lấy metadata của video (thông tin về video)
            meta = reader.get_meta_data()
            # Lấy FPS (frames per second) của video, mặc định 30 nếu không có
            fps = meta.get('fps', 30)
            # Tính thời gian delay giữa các frame để đồng bộ với FPS gốc
            frame_delay = 1 / fps
            
            self.logger.info(f"Video FPS: {fps}, Kích thước: {meta.get('size', 'N/A')}")

            # Chỉ xử lý mỗi 3 frame để tối ưu hiệu năng (giảm tải xử lý)
            process_every_n_frames = 3
            # Lưu trữ các phát hiện khuôn mặt và cảm xúc của frame trước đó
            # Để vẽ lên các frame không được xử lý
            last_detections = []

            # Duyệt qua từng frame trong video
            for i, frame in enumerate(reader):
                # Ghi nhận thời gian bắt đầu xử lý frame để đồng bộ FPS
                start_time = time.time()
                # imageio đọc frame ở định dạng RGB, chuyển sang BGR cho OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Lưu frame gốc để hiển thị debug
                original_frame = frame_bgr.copy()
                # Khởi tạo danh sách cảm xúc cho frame này
                frame_emotions = []

                # Chỉ chạy phát hiện khuôn mặt trên một số frame nhất định (mỗi 3 frame)
                # Điều này giúp tăng tốc độ xử lý mà vẫn giữ được độ mượt
                if i % process_every_n_frames == 0:
                    # Chuyển frame sang grayscale để phát hiện khuôn mặt
                    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    # Phát hiện khuôn mặt trong frame
                    faces = self.facec.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    # Khởi tạo danh sách phát hiện hiện tại
                    current_detections = []
                    
                    # *** THÊM: Lưu trữ ROI và predictions để hiển thị debug ***
                    rois_list = []
                    predictions_list = []
                    
                    # Log khi phát hiện khuôn mặt
                    if len(faces) > 0:
                        self.logger.info(f"Frame {i}: Phát hiện {len(faces)} khuôn mặt")
                    
                    # Xử lý từng khuôn mặt tìm được
                    for (x, y, w, h) in faces:
                        # Cắt vùng khuôn mặt từ ảnh grayscale
                        fc = gray_frame[y:y+h, x:x+w]
                        # Resize về 48x48 pixels
                        roi = cv2.resize(fc, (48, 48))
                        # Dự đoán cảm xúc
                        pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                        # Thêm cảm xúc vào danh sách
                        frame_emotions.append(pred)
                        # Lưu vị trí và cảm xúc của khuôn mặt này
                        current_detections.append((x, y, w, h, pred))
                        
                        # *** THÊM: Lưu ROI và prediction ***
                        rois_list.append((roi.copy(), pred))
                        predictions_list.append(pred)
                    
                    # Cập nhật danh sách phát hiện cuối cùng
                    last_detections = current_detections
                    
                    # *** THÊM: Lưu frame để hiển thị debug sau này ***
                    if len(faces) > 0 and len(predictions_list) > 0:
                        frame_data = (
                            original_frame.copy(),
                            gray_frame.copy(),
                            faces.copy(),
                            rois_list.copy(),
                            predictions_list.copy(),
                            frame_bgr.copy()
                        )
                        collected_frames.append(frame_data)
                
                # Vẽ các phát hiện cuối cùng lên frame hiện tại (vẽ lên mọi frame)
                # Điều này đảm bảo video mượt mà, không bị giật
                for (x, y, w, h, pred) in last_detections:
                    # Vẽ khung bao quanh khuôn mặt
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    # Vẽ text cảm xúc
                    cv2.putText(frame_bgr, pred, (x, y-10), self.font, 0.9, (255, 255, 0), 2)

                # Encode frame thành định dạng JPEG để gửi qua mạng
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                # Chuyển buffer thành bytes
                frame_bytes = buffer.tobytes()
                # Yield frame và cảm xúc (generator function - trả về từng frame)
                yield (frame_bytes, frame_emotions)

                # Đồng bộ với FPS gốc của video
                # Tính thời gian đã xử lý
                elapsed_time = time.time() - start_time
                # Tính thời gian cần sleep để đạt đúng FPS
                sleep_time = frame_delay - elapsed_time
                # Nếu còn thời gian, sleep để đồng bộ
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            # *** QUAN TRỌNG: Block finally đảm bảo debug window luôn hiển thị ***
            # Dù generator có chạy hết hay bị ngắt giữa chừng
            
            # *** THÊM: Sau khi xử lý xong video, hiển thị debug windows ***
            self.logger.info(f"Đã xử lý xong video, tổng số frame có khuôn mặt: {len(collected_frames)}")
            
            if len(collected_frames) > 0:
                self.logger.separator()
                self.logger.info(f"Đang hiển thị cửa sổ debug cho video...")
                
                # Nhóm các frame theo cảm xúc (giống webcam)
                emotion_groups = {}
                
                for frame_data in collected_frames:
                    original_frame, gray_frame, faces, rois, predictions, final_frame = frame_data
                    
                    for pred in predictions:
                        if pred not in emotion_groups:
                            filtered_rois = [(roi, emotion) for roi, emotion in rois if emotion == pred]
                            emotion_groups[pred] = {
                                'frame_data': (original_frame, gray_frame, faces, filtered_rois, [pred], final_frame),
                                'count': 1
                            }
                        else:
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
                    
                    self.debug_visualizer.show_processing_steps(
                        original_img=original_frame,
                        gray_img=gray_frame,
                        faces=faces,
                        rois=filtered_rois,
                        predictions=predictions,
                        final_img=final_frame
                    )
                
                self.logger.success(f"Đã hiển thị {len(emotion_groups)} cửa sổ debug (1 cửa sổ/cảm xúc)")
                
                # In thống kê
                self.logger.separator()
                self.logger.info("THỐNG KÊ CẢM XÚC VIDEO:")
                total_detections = sum(data['count'] for data in emotion_groups.values())
                for emotion, data in sorted(emotion_groups.items(), key=lambda x: x[1]['count'], reverse=True):
                    count = data['count']
                    percentage = (count / total_detections) * 100
                    self.logger.info(f"  • {emotion}: {count} lần ({percentage:.1f}%)")
                self.logger.separator()
            
            self.logger.end_processing("XỬ LÝ VIDEO STREAMING")

    def process_video(self, video_path, output_path):
        """
        Xử lý video file và lưu video đã xử lý ra file mới
        Args:
            video_path: Đường dẫn video đầu vào
            output_path: Đường dẫn video đầu ra
        Returns:
            output_path: Đường dẫn video đã xử lý
            emotion_counts: Dictionary đếm cảm xúc
        """
        # *** THÊM: Lưu trữ dữ liệu để hiển thị debug sau khi xử lý xong ***
        collected_frames = []
        
        # Bắt đầu logging
        self.logger.start_processing("XỬ LÝ VIDEO FILE", video_path)
        
        print(f"Opening video with imageio: {video_path}")
        try:
            # Mở video bằng imageio
            reader = imageio.get_reader(video_path)
            # Lấy metadata
            meta = reader.get_meta_data()
            # Lấy FPS, mặc định 30 nếu không tìm thấy
            fps = meta.get('fps', 30)
            print(f"Video properties (from imageio): {meta.get('size', 'N/A')}, {fps} FPS")
            self.logger.info(f"Video FPS: {fps}, Kích thước: {meta.get('size', 'N/A')}")
        except Exception as e:
            # Nếu không mở được, trả về lỗi
            print(f"ERROR: Cannot open video with imageio: {e}")
            self.logger.error(f"Không thể mở video: {e}")
            return None, {"error": f"Không thể mở file video: {e}"}

        # Tạo writer để ghi video đầu ra với cùng FPS
        # imageio tự động xử lý codec
        writer = imageio.get_writer(output_path, fps=fps)

        # Khởi tạo dictionary đếm cảm xúc
        emotion_counts = {
            "Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0,
            "Neutral": 0, "Sad": 0, "Surprise": 0
        }
        
        # Biến đếm số frame
        frame_count = 0
        # Chỉ xử lý mỗi 5 frame (tối ưu hiệu năng)
        process_every_n_frames = 5
        
        # Duyệt qua từng frame
        for i, frame in enumerate(reader):
            frame_count = i + 1
            # imageio đọc frame ở RGB, chuyển sang BGR cho OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Lưu frame gốc
            original_frame = frame_bgr.copy()

            # Chỉ phát hiện khuôn mặt mỗi 5 frame
            if frame_count % process_every_n_frames == 0:
                # Chuyển sang grayscale
                gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                # Phát hiện khuôn mặt
                faces = self.facec.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # *** THÊM: Lưu trữ ROI và predictions ***
                rois_list = []
                predictions_list = []
                
                # Log khi phát hiện khuôn mặt
                if len(faces) > 0:
                    self.logger.info(f"Frame {frame_count}: Phát hiện {len(faces)} khuôn mặt")
                
                # Xử lý từng khuôn mặt
                for (x, y, w, h) in faces:
                    # Cắt ROI
                    fc = gray_frame[y:y+h, x:x+w]
                    # Resize về 48x48
                    roi = cv2.resize(fc, (48, 48))
                    # Dự đoán cảm xúc
                    pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                    # Đếm cảm xúc
                    emotion_counts[pred] += 1
                    
                    # *** THÊM: Lưu ROI và prediction ***
                    rois_list.append((roi.copy(), pred))
                    predictions_list.append(pred)
                    
                    # Vẽ khung
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    # Vẽ text
                    cv2.putText(frame_bgr, pred, (x, y-10), self.font, 0.9, (255, 255, 0), 2)
                
                # *** THÊM: Lưu frame để hiển thị debug sau này ***
                if len(faces) > 0 and len(predictions_list) > 0:
                    frame_data = (
                        original_frame.copy(),
                        gray_frame.copy(),
                        faces.copy(),
                        rois_list.copy(),
                        predictions_list.copy(),
                        frame_bgr.copy()
                    )
                    collected_frames.append(frame_data)
            
            # Chuyển frame từ BGR về RGB cho imageio writer (imageio dùng RGB)
            frame_rgb_processed = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Ghi frame vào video đầu ra
            writer.append_data(frame_rgb_processed)

            # In tiến trình mỗi 30 frame
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        # Đóng reader và writer
        reader.close()
        writer.close()
        
        # In thông tin hoàn thành
        print(f"Video processing complete! Total frames: {frame_count}")
        print(f"Final emotion counts: {emotion_counts}")
        
        # Cảnh báo nếu không phát hiện được khuôn mặt nào
        if sum(emotion_counts.values()) == 0:
            print("WARNING: No faces detected in the video!")
            self.logger.warning("Không phát hiện được khuôn mặt nào trong video!")
        
        # *** THÊM: Sau khi xử lý xong, hiển thị debug windows ***
        self.logger.info(f"Đã xử lý xong {frame_count} frame, frame có khuôn mặt: {len(collected_frames)}")
        
        if len(collected_frames) > 0:
            self.logger.separator()
            self.logger.info(f"Đang hiển thị cửa sổ debug cho video...")
            
            # Nhóm các frame theo cảm xúc
            emotion_groups = {}
            
            for frame_data in collected_frames:
                original_frame, gray_frame, faces, rois, predictions, final_frame = frame_data
                
                for pred in predictions:
                    if pred not in emotion_groups:
                        filtered_rois = [(roi, emotion) for roi, emotion in rois if emotion == pred]
                        emotion_groups[pred] = {
                            'frame_data': (original_frame, gray_frame, faces, filtered_rois, [pred], final_frame),
                            'count': 1
                        }
                    else:
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
                
                self.debug_visualizer.show_processing_steps(
                    original_img=original_frame,
                    gray_img=gray_frame,
                    faces=faces,
                    rois=filtered_rois,
                    predictions=predictions,
                    final_img=final_frame
                )
            
            self.logger.success(f"Đã hiển thị {len(emotion_groups)} cửa sổ debug (1 cửa sổ/cảm xúc)")
            
            # In thống kê
            self.logger.separator()
            self.logger.info("THỐNG KÊ CẢM XÚC VIDEO:")
            total_detections = sum(data['count'] for data in emotion_groups.values())
            for emotion, data in sorted(emotion_groups.items(), key=lambda x: x[1]['count'], reverse=True):
                count = data['count']
                percentage = (count / total_detections) * 100
                self.logger.info(f"  • {emotion}: {count} lần ({percentage:.1f}%)")
            self.logger.separator()
        
        # Kết thúc logging
        self.logger.end_processing("XỬ LÝ VIDEO FILE", emotion_counts)
        
        # Trả về đường dẫn video và thống kê cảm xúc
        return output_path, emotion_counts
