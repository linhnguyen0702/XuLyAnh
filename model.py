# Import hàm để load model từ file JSON
from tensorflow.keras.models import model_from_json
# Import hàm để set TensorFlow session
from tensorflow.python.keras.backend import set_session
# Import numpy để xử lý mảng và tìm index max
import numpy as np

# Import TensorFlow
import tensorflow as tf

# Cấu hình TensorFlow session
# ConfigProto: cấu hình cho TensorFlow session
config = tf.compat.v1.ConfigProto()
# Giới hạn sử dụng GPU memory: chỉ dùng 15% bộ nhớ GPU
# Tránh chiếm hết GPU memory, để lại cho các ứng dụng khác
config.gpu_options.per_process_gpu_memory_fraction = 0.15
# Tạo TensorFlow session với config đã đặt
session = tf.compat.v1.Session(config=config)
# Set session này làm session mặc định cho Keras backend
set_session(session)


class FacialExpressionModel(object):
    """
    Class để load và sử dụng model CNN nhận diện cảm xúc
    Model được train để phân loại 7 loại cảm xúc
    """

    # Danh sách 7 cảm xúc mà model có thể nhận diện
    # Thứ tự này phải khớp với thứ tự output của model
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        """
        Khởi tạo model từ file JSON và weights
        Args:
            model_json_file: Đường dẫn file JSON chứa kiến trúc model
            model_weights_file: Đường dẫn file H5 chứa trọng số đã train
        """
        # Bước 1: Load kiến trúc model từ file JSON
        # Mở file JSON ở chế độ đọc
        with open(model_json_file, "r") as json_file:
            # Đọc toàn bộ nội dung file JSON
            loaded_model_json = json_file.read()
            # Tạo model từ JSON string
            # model_from_json() khôi phục kiến trúc mạng neural network
            self.loaded_model = model_from_json(loaded_model_json)

        # Bước 2: Load trọng số (weights) đã được train vào model
        # load_weights() nạp các giá trị trọng số từ file H5
        # Đây là các giá trị đã được train sẵn, không cần train lại
        self.loaded_model.load_weights(model_weights_file)
        # Các dòng dưới đây đã bị comment vì không cần thiết
        #self.loaded_model.compile()  # Compile model (không cần nếu chỉ dùng để predict)
        #self.loaded_model._make_predict_function()  # Tạo predict function (tự động trong TF 2.x)

    def predict_emotion(self, img):
        """
        Dự đoán cảm xúc từ ảnh khuôn mặt
        Args:
            img: Ảnh đầu vào, shape (1, 48, 48, 1)
                 - 1: batch size (1 ảnh)
                 - 48, 48: kích thước ảnh (height, width)
                 - 1: số channel (grayscale)
        Returns:
            Tên cảm xúc có xác suất cao nhất (string)
        """
        # Lấy global session để đảm bảo dùng đúng session
        global session
        # Set session cho Keras backend
        set_session(session)
        
        # Bước 1: Dự đoán bằng model
        # model.predict() trả về mảng xác suất cho 7 cảm xúc
        # Shape output: (1, 7) - 1 ảnh, 7 xác suất
        # Ví dụ: [[0.1, 0.05, 0.02, 0.8, 0.01, 0.01, 0.01]]
        #         [Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise]
        self.preds = self.loaded_model.predict(img)
        
        # Bước 2: Tìm cảm xúc có xác suất cao nhất
        # np.argmax() trả về index của giá trị lớn nhất trong mảng
        # Ví dụ: np.argmax([0.1, 0.05, 0.02, 0.8, 0.01, 0.01, 0.01]) = 3
        # EMOTIONS_LIST[3] = "Happy"
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
