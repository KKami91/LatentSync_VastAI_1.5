import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import base64
from typing import Sequence, Mapping, Any, Union
from io import BytesIO
import glob
import argparse
import logging

#로컬 전용
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    #filename="/workspace/latent_sync.log",  # 로그 파일 경로
    filename="./latent_sync.log",  # 로그 파일 경로
    filemode="w",  # 파일 모드 (w: 덮어쓰기, a: 이어쓰기)
)
logger = logging.getLogger(__name__)

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
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = "/workspace/ComfyUI"
    if os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


# utils 디렉토리가 있는지 확인하고 __init__.py 생성
def check_utils_package():
    utils_dir = os.path.join("/workspace/ComfyUI", "utils")
    init_file = os.path.join(utils_dir, "__init__.py")
    
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        logger.info(f"Created utils directory at {utils_dir}")
    
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass  # 빈 __init__.py 파일 생성
        logger.info(f"Created __init__.py at {init_file}")

def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    logger.info("Adding extra model paths...")
    # utils.extra_config import 부분 제거하고 단순화
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        logger.info(f"Found extra_model_paths at: {extra_model_paths}")
    else:
        logger.info("No extra_model_paths.yaml found, skipping...")


# add_comfyui_directory_to_sys_path()
# add_extra_model_paths()

def setup_utils_package():
    """Setup utils directory as a Python package"""
    logger.info("Setting up utils package...")
    utils_dir = "/workspace/ComfyUI/utils"
    
    # utils 디렉토리가 없으면 생성
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        logger.info(f"Created utils directory at {utils_dir}")
    
    # __init__.py 파일 생성
    init_file = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass
        logger.info("Created __init__.py in utils directory")

    # json_util.py가 없으면 생성
    json_util_file = os.path.join(utils_dir, "json_util.py")
    if not os.path.exists(json_util_file):
        with open(json_util_file, 'w') as f:
            f.write("""
def merge_json_recursive(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_json_recursive(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1
""")
        logger.info("Created json_util.py in utils directory")


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




# def setup_environment():
#     """Setup the ComfyUI environment"""
#     add_comfyui_directory_to_sys_path()
#     add_extra_model_paths()

#     import folder_paths
#     custom_node_paths = folder_paths.get_folder_paths("custom_nodes")
#     logger.info("custom_node_paths : ", custom_node_paths)

#     import_custom_nodes()

def setup_environment():
    """Setup the ComfyUI environment"""
    logger.info("Setting up environment...")
    
    # ComfyUI 디렉토리를 Python 경로에 추가
    comfyui_path = "/workspace/ComfyUI"
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
        logger.info(f"Added {comfyui_path} to sys.path")

    # utils 디렉토리가 패키지로 인식되도록 __init__.py 확인
    utils_init = os.path.join(comfyui_path, "utils", "__init__.py")
    if not os.path.exists(utils_init):
        with open(utils_init, 'w') as f:
            pass
        logger.info("Created __init__.py in utils directory")

    # Python path 로깅
    logger.info("Python path:")
    for path in sys.path:
        logger.info(f"  - {path}")
    
    try:
        # 모듈 import 테스트
        logger.info("Testing imports...")
        import utils.json_util as json_util
        logger.info("Successfully imported json_util")
        
        # Custom nodes 초기화
        import_custom_nodes()
        logger.info("Custom nodes initialized successfully")
    except Exception as e:
        logger.error(f"Error during environment setup: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir('.')}")
        if os.path.exists("/workspace/ComfyUI/utils"):
            logger.error(f"Utils directory contents: {os.listdir('/workspace/ComfyUI/utils')}")
        raise


def process_latentsync(video_data: bytes, audio_data: bytes, video_name: str, custom_width_: int, custom_height_: int):
    from nodes import NODE_CLASS_MAPPINGS
    import os
    import tempfile

    logger.info("innnnn pppppppprocess_latentsynccccccccccccccccc")

    setup_environment()

    logger.info("setup environment 진행 후................")

    video_name_without_ext = os.path.splitext(video_name)[0]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 입력 파일 설정
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            output_filename = f"convert_{video_name_without_ext}"
            
            # 입력 파일 저장
            with open(video_path, "wb") as f:
                f.write(video_data)
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            logger.info("Starting LatentSync processing...")
            
            try:
                with torch.inference_mode():
                    # LoadAudio
                    loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
                    loadaudio_37 = loadaudio.load(audio=audio_path)

                    # LoadVideo
                    vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
                    vhs_loadvideo_40 = vhs_loadvideo.load_video(
                        video=video_path,
                        force_rate=25,
                        custom_width=custom_width_,
                        custom_height=custom_height_,
                        frame_load_cap=0,
                        skip_first_frames=0,
                        select_every_nth=1,
                        format="AnimateDiff",
                        unique_id=12015943199208297010,
                    )

                    d_videolengthadjuster = NODE_CLASS_MAPPINGS["D_VideoLengthAdjuster"]()
                    d_latentsyncnode = NODE_CLASS_MAPPINGS["D_LatentSyncNode"]()
                    vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                    
                    d_videolengthadjuster_53 = d_videolengthadjuster.adjust(
                        mode="pingpong",
                        fps=25,
                        silent_padding_sec=0.5,
                        images=get_value_at_index(vhs_loadvideo_40, 0),
                        audio=get_value_at_index(loadaudio_37, 0),
                    )

                    d_latentsyncnode_43 = d_latentsyncnode.inference(
                        seed=random.randint(1, 2**32 - 1),
                        images=get_value_at_index(d_videolengthadjuster_53, 0),
                        audio=get_value_at_index(d_videolengthadjuster_53, 1),
                    )

                    logger.info(f"Processing video combine with filename: {output_filename}")
                    
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
                        images=get_value_at_index(d_latentsyncnode_43, 0),
                        audio=get_value_at_index(d_latentsyncnode_43, 1),
                        unique_id=7599875590960303900,
                    )

                    # 결과에서 파일 경로 가져오기
                    if isinstance(result, dict) and 'result' in result:
                        saved_files = result['result'][0][1]
                    else:
                        saved_files = result[0][1]

                    if not saved_files:
                        raise Exception("No output files were generated")

                    result_path = saved_files[-1]  # 마지막 파일이 최종 결과물
                    logger.info(f"Result file path: {result_path}")

                    if not os.path.exists(result_path):
                        raise FileNotFoundError(f"Output file not found at: {result_path}")

                    with open(result_path, "rb") as f:
                        output_data = f.read()

                    logger.info("Successfully processed video")
                    
                    return {
                        "output": {
                            "video_data": base64.b64encode(output_data).decode('utf-8'),
                            "video_name": f"{output_filename}.mp4"
                        }
                    }

            except Exception as e:
                logger.error(f"Error during LatentSync processing: {str(e)}")
                return {"error": str(e)}

        except Exception as e:
            logger.error(f"Error in file handling: {str(e)}")
            return {"error": str(e)}
        

        # finally:
        #     print('in finally....')
        #     temp_path = os.path.dirname(result_path)
        #     comfyui_path = os.path.abspath(os.path.join(temp_path, ".."))
        #     if os.path.exists(temp_path):
        #         print('.mp4 제거..?')
        #         [os.remove(mp4_file) for mp4_file in glob.glob(os.path.join(temp_path, "*.mp4"))]
        #     if os.path.exists(comfyui_path):
        #         print('.wav 제거..?')aaaaa
        #         [os.remove(wav_file) for wav_file in glob.glob(os.path.join(comfyui_path, "*.wav"))]

            

def handler(event):
    import os
    import glob

    print('handler 시작?')
    try:
        # 입력 데이터 검증
        if 'input' not in event or 'video' not in event['input'] or 'audio' not in event['input']:
            raise ValueError("Missing required input fields (video and/or audio)")

        # 입력 데이터 디코딩
        video_data = base64.b64decode(event['input']['video'])
        audio_data = base64.b64decode(event['input']['audio'])
        video_name = event['input']['video_name']
        custom_width_ = event['input']['custom_width_']
        custom_height_ = event['input']['custom_height_']       
        # 환경 설정
        setup_environment()
        
        # 처리
        logging.info("process_latentsync 전....!")
        result = process_latentsync(video_data, audio_data, video_name, custom_width_, custom_height_)
        

        

        
        print("Cleanup completed")
        return result
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}
    finally:
        # 추가적인 cleanup이 필요한 경우 여기서 처리
                # Cleanup 처리
        print("Cleaning up temporary files...")
        temp_dir = os.path.join(os.getcwd(), "temp")
        output_dir = os.path.join(os.getcwd(), "output")
        
        # 임시 파일들 정리
        cleanup_patterns = [
            os.path.join(temp_dir, "*.mp4"),
            os.path.join(temp_dir, "*.wav"),
            os.path.join(output_dir, "*.mp4"),
            os.path.join(output_dir, "*.wav"),
            os.path.join(os.getcwd(), "*.wav"),  # 루트 디렉토리의 wav 파일
        ]
        for pattern in cleanup_patterns:
            try:
                files = glob.glob(pattern)
                for file in files:
                    try:
                        os.remove(file)
                        print(f"Removed: {file}")
                    except Exception as e:
                        print(f"Error removing {file}: {str(e)}")
            except Exception as e:
                print(f"Error processing pattern {pattern}: {str(e)}")
        print("Handler completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LatentSync")
    parser.add_argument("--video", required=True, help="Input Video")
    parser.add_argument("--audio", required=True, help="Input Audio")
    parser.add_argument("--output", required=True, help="Output Video")
    parser.add_argument("--width", type=int, default=0, help="Custom Width")
    parser.add_argument("--height", type=int, default=0, help="Custom Height")

    args = parser.parse_args()
    with open(args.video, 'rb') as f:
        video_data = f.read()
    with open(args.audio, 'rb') as f:
        audio_data = f.read()
    video_name = os.path.basename(args.video)


    logger.info("LatentSync 작업 시작")

    try:
        result = process_latentsync(video_data, audio_data, video_name, args.width, args.height)

        output_video_data = base64.b64decode(result['output']['video_data'])

        with open(args.output, 'wb') as f:
            f.write(output_video_data)
        
        logger.info(f"LatentSync 완료. 결과 파일: {args.output}")

    except Exception as e:
        logger.exception("LatentSync 작업 중 오류")
