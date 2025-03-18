import os
import random
import sys
import base64
import torch
import tempfile
import glob
import logging
from typing import Sequence, Mapping, Any, Union
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import cv2
import numpy as np
import subprocess
import uuid  # 고유 ID 생성을 위해 추가
import threading
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# GPU 관리를 위한 클래스
class GPUManager:
    def __init__(self, max_tasks_per_gpu=2):
        self.num_gpus = torch.cuda.device_count()
        self.gpu_locks = [threading.Lock() for _ in range(self.num_gpus)]
        # 각 GPU의 현재 작업 수를 추적합니다
        self.gpu_task_count = [0 for _ in range(self.num_gpus)]
        # GPU당 최대 작업 수
        self.max_tasks_per_gpu = max_tasks_per_gpu
        self.manager_lock = threading.Lock()
        logger.info(f"Initialized GPU Manager with {self.num_gpus} GPUs (max {max_tasks_per_gpu} tasks per GPU)")

    def get_available_gpu(self):
        """작업 수가 가장 적은 사용 가능한 GPU 인덱스를 반환합니다."""
        with self.manager_lock:
            # 최소 작업 수를 가진 GPU 찾기
            min_tasks = self.max_tasks_per_gpu
            selected_gpu = None
            
            for i in range(self.num_gpus):
                if self.gpu_task_count[i] < min_tasks:
                    min_tasks = self.gpu_task_count[i]
                    selected_gpu = i
            
            # 최대 작업 수 제한 확인
            if selected_gpu is not None and self.gpu_task_count[selected_gpu] < self.max_tasks_per_gpu:
                self.gpu_task_count[selected_gpu] += 1
                logger.info(f"Allocated GPU {selected_gpu} (now running {self.gpu_task_count[selected_gpu]} tasks)")
                return selected_gpu
            
            # 모든 GPU가 최대 작업 수에 도달한 경우
            logger.warning("All GPUs are at maximum capacity. Waiting for an available GPU.")
            return None

    def wait_for_available_gpu(self, timeout=None):
        """사용 가능한 GPU가 생길 때까지 대기합니다."""
        start_time = time.time()
        while True:
            gpu_id = self.get_available_gpu()
            if gpu_id is not None:
                return gpu_id
            
            # 타임아웃 체크
            if timeout and (time.time() - start_time > timeout):
                raise TimeoutError("Timed out waiting for an available GPU")
            
            # 잠시 대기 후 다시 시도
            time.sleep(1)

    def release_gpu(self, gpu_id):
        """GPU의 작업 카운트를 감소시킵니다."""
        with self.manager_lock:
            if 0 <= gpu_id < self.num_gpus and self.gpu_task_count[gpu_id] > 0:
                self.gpu_task_count[gpu_id] -= 1
                logger.info(f"Released GPU {gpu_id} (now running {self.gpu_task_count[gpu_id]} tasks)")
            else:
                logger.warning(f"Attempted to release invalid GPU ID: {gpu_id} or already at 0 tasks")

    def with_gpu(self, gpu_id):
        """특정 GPU를 사용하는 컨텍스트 매니저를 반환합니다."""
        class GPUContext:
            def __init__(self, manager, gpu_id):
                self.manager = manager
                self.gpu_id = gpu_id
                
            def __enter__(self):
                return self.gpu_id
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.manager.release_gpu(self.gpu_id)
                
        return GPUContext(self, gpu_id)

    def get_status(self):
        """각 GPU의 현재 상태를 반환합니다."""
        with self.manager_lock:
            status = []
            for i in range(self.num_gpus):
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    free_memory = total_memory - allocated_memory
                    
                    status.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "tasks_running": self.gpu_task_count[i],
                        "max_tasks": self.max_tasks_per_gpu,
                        "total_memory_gb": round(total_memory / (1024**3), 2),
                        "used_memory_gb": round(allocated_memory / (1024**3), 2),
                        "free_memory_gb": round(free_memory / (1024**3), 2)
                    })
                except Exception as e:
                    status.append({
                        "id": i,
                        "name": "Unknown",
                        "tasks_running": self.gpu_task_count[i],
                        "max_tasks": self.max_tasks_per_gpu,
                        "error": str(e)
                    })
            return status

# GPU 매니저 인스턴스 생성 - GPU당 최대 2개 작업 허용
gpu_manager = GPUManager(max_tasks_per_gpu=2)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add 'ComfyUI' to the sys.path"""
    comfyui_path = "/workspace/ComfyUI"
    if os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        logger.info(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path."""
    try:
        from main import load_extra_path_config
    except ImportError:
        logger.info("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        logger.info("Could not find the extra_model_paths config file.")


def setup_environment():
    """Setup the ComfyUI environment"""
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


def process_latentsync(video_data: bytes, audio_data: bytes, video_name: str, custom_width_: int, 
                       custom_height_: int, force_rate: int, fps: float, lips_expression_: float, 
                       inference_steps_: int):
    from nodes import NODE_CLASS_MAPPINGS

    # 사용 가능한 GPU 할당 받기
    gpu_id = gpu_manager.wait_for_available_gpu(timeout=600)  # 10분 타임아웃

    try:
        # 현재 GPU 설정
        torch.cuda.set_device(gpu_id)
        logger.info(f"Processing on GPU {gpu_id}")
        
        # GPU 메모리 초기화
        torch.cuda.empty_cache()
        
        # CUDA 환경 변수 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 고유한 처리 ID 생성
        process_id = str(uuid.uuid4())
        video_name_without_ext = os.path.splitext(video_name)[0]
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 고유한 파일 이름 생성
                video_path = os.path.join(temp_dir, f"{process_id}_input_video.mp4")
                audio_path = os.path.join(temp_dir, f"{process_id}_input_audio.wav")
                output_filename = f"convert_{video_name_without_ext}_{process_id}"
                
                # 입력 파일 저장
                with open(video_path, "wb") as f:
                    f.write(video_data)
                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                logger.info("Starting LatentSync processing...")

                try:
                    with torch.inference_mode():
                        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
                        loadaudio_37 = loadaudio.load(audio=audio_path)

                        # LoadVideo
                        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
                        vhs_loadvideo_40 = vhs_loadvideo.load_video(
                            video=video_path,
                            #force_rate=25,
                            force_rate=force_rate,
                            #custom_width=custom_width_,
                            #custom_height=custom_height_,
                            custom_width=0,
                            custom_height=0,
                            frame_load_cap=0,
                            skip_first_frames=0,
                            select_every_nth=1,
                            format="AnimateDiff",
                            unique_id=12015943199208297010,
                        )

                        videolengthadjuster = NODE_CLASS_MAPPINGS["VideoLengthAdjuster"]()
                        latentsyncnode = NODE_CLASS_MAPPINGS["LatentSyncNode"]()
                        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

                        videolengthadjuster_55 = videolengthadjuster.adjust(
                            mode="pingpong",
                            #fps=25,
                            fps=fps,
                            silent_padding_sec=0.5,
                            images=get_value_at_index(vhs_loadvideo_40, 0),
                            audio=get_value_at_index(loadaudio_37, 0),
                        )                                                            

                        try:
                            # LatentSync processing attempt
                            logger.info("Attempting LatentSync processing...")
                            
                            # Add code to prevent meta tensor error
                            torch.cuda.empty_cache()
                            
                            # Initialize result variable
                            result = None

                            try:
                                latentsyncnode_54 = latentsyncnode.inference(
                                    seed=random.randint(1, 2**32 - 1),
                                    lips_expression=lips_expression_, # 파라미터 수정 가능
                                    inference_steps=inference_steps_, # 파라미터 수정 가능
                                    images=get_value_at_index(videolengthadjuster_55, 0),
                                    audio=get_value_at_index(videolengthadjuster_55, 1),
                                )

                                logger.info(f"Processing successful, generating video with filename: {output_filename}")

                                result = vhs_videocombine.combine_video(
                                    frame_rate=25,
                                    loop_count=0,
                                    filename_prefix=output_filename,
                                    format="video/h264-mp4",
                                    pix_fmt="yuv420p",
                                    crf=19,
                                    save_metadata=True,
                                    trim_to_audio=False,
                                    pingpong=False,
                                    save_output=False,
                                    images=get_value_at_index(latentsyncnode_54, 0),
                                    audio=get_value_at_index(latentsyncnode_54, 1),
                                    unique_id=7599875590960303900,
                                )

                            except Exception as e:
                                    # Define error message variable
                                    error_msg = str(e)
                                    logger.warning(f"LatentSync processing error: {error_msg}")

                                    # Check if it's a face detection related error
                                    face_detection_error = any(phrase in error_msg for phrase in [
                                        "No face detected", 
                                        "expected Tensor", 
                                        "face detection failed", 
                                        "could not find face",
                                        "but got NoneType",
                                        "Cannot copy out of meta tensor",  # Add meta tensor error
                                        "No face detected in any frame",   # 추가 오류 메시지
                                        "얼굴 감지 실패",                  # 추가 오류 메시지
                                        "비디오의 모든 프레임에서 얼굴이 감지되지 않았습니다",  # 추가 오류 메시지
                                        "영상에서 얼굴을 찾을 수 없거나 일부 프레임에만 얼굴이 있습니다"  # 추가 오류 메시지
                                    ])              

                                    # Check for specific face detection error
                                    if face_detection_error:
                                        logger.info("얼굴 감지 실패: 비디오에서 충분한 얼굴이 감지되지 않았습니다. 원본 비디오에 오디오만 추가합니다.")
                                    else:
                                        logger.warning(f"예상치 못한 오류 발생: {error_msg}")

                                    # Combine original video and audio
                                    logger.info("대체 처리: 원본 비디오에 오디오만 추가합니다")
                                    result = vhs_videocombine.combine_video(
                                        frame_rate=25,
                                        loop_count=0,
                                        filename_prefix=output_filename,
                                        format="video/h264-mp4",
                                        pix_fmt="yuv420p",
                                        crf=19,
                                        save_metadata=True,
                                        trim_to_audio=False,
                                        pingpong=False,
                                        save_output=False,
                                        images=get_value_at_index(videolengthadjuster_55, 0),
                                        audio=get_value_at_index(videolengthadjuster_55, 1),
                                        unique_id=7599875590960303900,
                                    )
                            # Check if result is still None
                            if result is None:
                                raise Exception("Video processing failed: No result was generated")

                        except Exception as e:
                            logger.error(f"Error during LatentSync processing: {str(e)}")
                            return {"success": False, "error": str(e)}

                        # 결과에서 파일 경로 가져오기
                        if isinstance(result, dict) and 'result' in result:
                            saved_files = result['result'][0][1]
                        else:
                            saved_files = result[0][1]

                        if not saved_files:
                            raise Exception("Output file was not generated")

                        result_path = saved_files[-1]  # 마지막 파일이 최종 결과물
                        logger.info(f"Result file path: {result_path}")

                        if not os.path.exists(result_path):
                            raise FileNotFoundError(f"Output file not found at path: {result_path}")

                        # 결과 파일 영구 저장 (선택 사항)
                        permanent_output_path = os.path.join(output_dir, f"{output_filename}.mp4")
                        with open(result_path, "rb") as src, open(permanent_output_path, "wb") as dst:
                            dst.write(src.read())

                        with open(result_path, "rb") as f:
                            output_data = f.read()

                        logger.info("Video processing successful")

                        return {
                            "success": True,
                            "output": {
                                "video_data": base64.b64encode(output_data).decode('utf-8'),
                                "video_name": f"{output_filename}.mp4",
                                "file_path": permanent_output_path
                            }
                        }

                except Exception as e:
                    logger.error(f"Error during LatentSync processing: {str(e)}")
                    return {"success": False, "error": str(e)}

            except Exception as e:
                logger.error(f"Error during file processing: {str(e)}")
                return {"success": False, "error": str(e)}

            finally:
                # 임시 파일 정리
                cleanup_patterns = [
                    os.path.join(temp_dir, f"{process_id}_*.mp4"),
                    os.path.join(temp_dir, f"{process_id}_*.wav")
                ]
                
                for pattern in cleanup_patterns:
                    try:
                        files = glob.glob(pattern)
                        for file in files:
                            try:
                                os.remove(file)
                                logger.info(f"Deleted file: {file}")
                            except Exception as e:
                                logger.error(f"Error deleting {file}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing pattern {pattern}: {str(e)}")
                
                # 최종 GPU 메모리 정리
                torch.cuda.empty_cache()
    finally:
        # GPU 사용 해제
        gpu_manager.release_gpu(gpu_id)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "LatentSync API",
        "status": "running",
        "gpu_count": torch.cuda.device_count(),
        "usage": {
            "endpoint": "/latentsync",
            "method": "POST",
            "required_fields": ["video", "audio", "video_name"],
            "optional_fields": ["custom_width", "custom_height"]
        }
    })

@app.route('/latentsync', methods=['POST'])
def handle_latentsync():
    try:
        # API 요청에서 데이터 추출
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400
        
        data = request.json
        
        # 필수 필드 검증
        if 'video' not in data or 'audio' not in data or 'video_name' not in data:
            return jsonify({"success": False, "error": "Missing required fields (video, audio, video_name)"}), 400
        
        # 선택적 필드 설정
        custom_width = int(data.get('custom_width', 64))
        custom_height = int(data.get('custom_height', 64))
        
        # 데이터 디코딩
        try:
            video_data = base64.b64decode(data['video'])
            audio_data = base64.b64decode(data['audio'])
        except:
            return jsonify({"success": False, "error": "Invalid base64 encoding"}), 400
        

        lips_expression = float(data.get('lips_expression', 1.5))
        inference_steps = int(data.get('inference_steps', 20))
        
        # force_rate, fps 조절
        force_rate = int(data.get('force_rate', 25))
        fps = float(data.get('fps', 25.0))

        # 처리
        result = process_latentsync(
            video_data, 
            audio_data, 
            data['video_name'], 
            custom_width, 
            custom_height,
            force_rate,
            fps,
            lips_expression,
            inference_steps
        )
        
        # 에러 처리
        if not result.get('success', True):
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in API handler: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """파일 다운로드 엔드포인트"""
    try:
        output_dir = os.path.join(os.getcwd(), "output")
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "File not found"}), 404
            
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download handler: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_gpu_status():
    """GPU 상태 확인 엔드포인트"""
    try:
        gpu_status = gpu_manager.get_status()
        
        return jsonify({
            "success": True,
            "gpu_count": torch.cuda.device_count(),
            "max_tasks_per_gpu": gpu_manager.max_tasks_per_gpu,
            "gpus": gpu_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error in GPU status handler: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def patch_face_alignment():
    """face_alignment 라이브러리의 sfd_detector.py 파일을 패치합니다."""
    try:
        import face_alignment
        
        # 라이브러리 경로 찾기
        fa_path = os.path.dirname(face_alignment.__file__)
        detector_path = os.path.join(fa_path, "detection", "sfd", "sfd_detector.py")
        
        if not os.path.exists(detector_path):
            logger.warning(f"Cannot find sfd_detector.py at: {detector_path}")
            return False
        
        # 현재 파일 내용 읽기
        with open(detector_path, 'r') as f:
            current_content = f.read()
        
        # 이미 패치가 적용되었는지 확인
        if "try:\n            self.face_detector.to(device)" in current_content:
            logger.info("Face alignment library is already patched")
            return True
        
        # 백업 파일 경로
        backup_path = f"{detector_path}.backup"
        
        # 백업 파일이 없으면 현재 파일을 백업
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(current_content)
            logger.info(f"Created backup file: {backup_path}")
            
            # 원본 내용에 패치 적용
            content_to_patch = current_content
        else:
            # 백업 파일에서 원본 내용 가져오기
            with open(backup_path, 'r') as f:
                content_to_patch = f.read()
            logger.info(f"Using backup file for patching: {backup_path}")
        
        # to() 메서드 호출 부분 찾기 및 수정
        if "self.face_detector.to(device)" in content_to_patch:
            # 패치 적용
            modified_content = content_to_patch.replace(
                "self.face_detector.to(device)",
                """try:
            self.face_detector.to(device)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                # meta tensor 처리를 위한 to_empty 메서드 사용
                self.face_detector.to_empty(device=device)
            else:
                raise"""
            )
            
            # 수정된 내용 저장
            with open(detector_path, 'w') as f:
                f.write(modified_content)
            
            logger.info("Successfully patched face_alignment sfd_detector.py")
            return True
        else:
            logger.warning("Could not find target code in sfd_detector.py")
            return False
    except Exception as e:
        logger.error(f"Failed to patch face_alignment: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Patching face_alignment library...")
    if patch_face_alignment():
        logger.info("Face alignment library patch applied successfully")
    else:
        logger.warning("Failed to apply face alignment library patch")
    
    # 환경 설정
    logger.info("Setting up ComfyUI environment...")
    setup_environment()
    logger.info("Environment setup complete")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "temp"), exist_ok=True)
    
    # 서버 실행
    logger.info(f"Starting Flask server on port 5000 with {torch.cuda.device_count()} GPUs available...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
