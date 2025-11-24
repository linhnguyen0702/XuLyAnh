import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, Response, request, jsonify
from camera import VideoCamera
from image_processor import ImageProcessor
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Tăng giới hạn lên 100 MB

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Configuration ---
# Bật/tắt hiển thị cửa sổ debug (hiển thị từng bước xử lý ảnh)
# Set True để hiển thị cửa sổ debug, False để tắt
SHOW_DEBUG_WINDOW = True  # Bật cửa sổ OpenCV debug để hiển thị từng bước xử lý

# Bật/tắt in log chi tiết ra terminal/console
# Set True để in log ra terminal, False để tắt
ENABLE_TERMINAL_LOG = True  # Bật log terminal

# --- Singleton Instances (Initialized Once) ---
camera = None
image_processor = None

try:
    # Khởi tạo image processor một lần duy nhất
    # Truyền show_debug_window để bật/tắt debug window
    # Truyền enable_terminal_log để bật/tắt log terminal
    image_processor = ImageProcessor(show_debug_window=SHOW_DEBUG_WINDOW, enable_terminal_log=ENABLE_TERMINAL_LOG)
    print("Model đã được tải thành công!")
    if SHOW_DEBUG_WINDOW:
        print("Debug window đã được bật - sẽ hiển thị từng bước xử lý ảnh")
    if ENABLE_TERMINAL_LOG:
        print("Terminal log đã được bật - sẽ in chi tiết quá trình xử lý ra terminal")
except Exception as e:
    print(f"Lỗi khởi tạo model: {e}")

# Khởi tạo camera sẵn để tăng tốc độ lần đầu
try:
    print("Đang khởi tạo camera...")
    # Truyền show_debug_window để bật/tắt debug window
    # Truyền enable_terminal_log để bật/tắt log terminal
    camera = VideoCamera(show_debug_window=SHOW_DEBUG_WINDOW, enable_terminal_log=ENABLE_TERMINAL_LOG)
    print("Camera đã sẵn sàng!")
except Exception as e:
    print(f"Không thể khởi tạo camera ngay: {e}")
    print("Camera sẽ được khởi tạo khi cần thiết.")
    camera = None

# Biến kiểm soát trạng thái và dữ liệu
camera_active = False
emotion_data = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0}
video_emotion_data = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0}

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

# --- Webcam Streaming ---
def gen(cam):
    """Webcam streaming generator function."""
    global camera_active, emotion_data
    while camera_active:
        frame_bytes, emotions = cam.get_frame()
        for emotion in emotions:
            if emotion in emotion_data:
                emotion_data[emotion] += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Endpoint để khởi tạo webcam trước khi streaming."""
    global camera, camera_active
    try:
        if camera is None:
            print("Đang khởi tạo camera...")
            # Truyền show_debug_window để bật/tắt debug window
            # Truyền enable_terminal_log để bật/tắt log terminal
            camera = VideoCamera(show_debug_window=SHOW_DEBUG_WINDOW, enable_terminal_log=ENABLE_TERMINAL_LOG)
            print("Camera đã sẵn sàng!")
        else:
            # Reset collected frames khi start lại webcam
            camera.reset_collected_frames()
        camera_active = True
        return jsonify({'success': True, 'message': 'Camera đã sẵn sàng'})
    except Exception as e:
        print(f"Lỗi khởi tạo camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Webcam streaming route."""
    global camera
    if camera is None or not camera_active:
        return jsonify({'success': False, 'error': 'Camera chưa được khởi tạo hoặc không hoạt động'}), 400
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Endpoint để giải phóng webcam."""
    global camera, camera_active, emotion_data
    camera_active = False
    if camera is not None:
        try:
            camera.release()
            camera = None
            print("Camera đã được giải phóng")
        except Exception as e:
            print(f"Lỗi khi giải phóng camera: {e}")
    emotion_data = {key: 0 for key in emotion_data}
    return jsonify({'success': True, 'message': 'Webcam đã dừng'})

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """Endpoint để lấy dữ liệu emotion từ webcam."""
    return jsonify({'success': True, 'emotions': emotion_data})

# --- Video Analysis Streaming ---
def video_gen(filename):
    """Video analysis streaming generator function."""
    global video_emotion_data
    video_emotion_data = {key: 0 for key in video_emotion_data} # Reset data
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        for frame_bytes, emotions in image_processor.process_video_stream(filepath):
            for emotion in emotions:
                if emotion in video_emotion_data:
                    video_emotion_data[emotion] += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    finally:
        # Clean up the uploaded file after streaming is done or if an error occurs
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Cleaned up uploaded video: {filepath}")
            except OSError as e:
                print(f"Error cleaning up file {filepath}: {e}")

@app.route('/video_analysis_feed/<filename>')
def video_analysis_feed(filename):
    """Route to stream processed video frames."""
    return Response(video_gen(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_emotions', methods=['GET'])
def get_video_emotions():
    """Endpoint to get emotion data from video analysis."""
    return jsonify({'success': True, 'emotions': video_emotion_data})

# --- File-based Analysis ---

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'File không hợp lệ'})
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    allowed_extensions = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif']
    if ext not in allowed_extensions:
        return jsonify({'success': False, 'error': f'File phải là ảnh. Nhận được: {ext}'})
    
    try:
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Xử lý ảnh và nhận kết quả
        result = image_processor.process_image(filepath, output_path)
        
        # Kiểm tra kết quả
        if result is None:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Không thể xử lý ảnh'})
        
        output_path_result, emotion_counts = result
        
        # Kiểm tra nếu có lỗi trong emotion_counts
        if isinstance(emotion_counts, dict) and 'error' in emotion_counts:
            os.remove(filepath)
            return jsonify({'success': False, 'error': emotion_counts['error']})
        
        # Đảm bảo emotion_counts là dictionary hợp lệ
        if not isinstance(emotion_counts, dict):
            emotion_counts = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, 
                            "Neutral": 0, "Sad": 0, "Surprise": 0}
        
        # Xóa file gốc
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'success': True, 'filename': output_filename, 'emotions': emotion_counts})
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Xóa file nếu có lỗi
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        return jsonify({'success': False, 'error': f'Lỗi xử lý ảnh: {str(e)}'})

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Handles video upload, saves it, and returns the filename for streaming."""
    file = request.files.get('video')
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'File không hợp lệ'})

    ext = file.filename.rsplit('.', 1)[1].lower()
    allowed_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
    if ext not in allowed_extensions:
        return jsonify({'success': False, 'error': f'File phải là video. Nhận được: {ext}'})

    try:
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Just return the filename, the client will use it to start the stream
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi lưu video: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
