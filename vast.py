import os
import sys
import torch
import random
import logging
import argparse
from typing import Sequence, Mapping, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="./latent_sync.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def setup_environment():
    """ComfyUI 환경 설정"""
    logger.info(".....setup environment 내부.....")
    logger.info("--------> ComfyUI/app/custom_node_manager.py ----> sys.path : ", sys.path)
    logger.info("sys.path???? ", sys.path)

    # Custom nodes 초기화
    try:
        logger.info('execution 전.......')
        import execution
        logger.info('server 전.......')
        import server
        logger.info('nodes 전.......')
        from nodes import NODE_CLASS_MAPPINGS, init_extra_nodes
        logger.info('asycio 전.......')
        import asyncio

        logger.info('loop 전.......')

        # 이벤트 루프 설정
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 서버 초기화
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)
        
        # 커스텀 노드 초기화
        init_extra_nodes()
        
        logger.info("Available nodes: %s", list(NODE_CLASS_MAPPINGS.keys()))
        
        required_nodes = ["LoadAudio", "VHS_LoadVideo", "D_VideoLengthAdjuster", 
                         "D_LatentSyncNode", "VHS_VideoCombine"]
        
        for node in required_nodes:
            if node not in NODE_CLASS_MAPPINGS:
                raise ImportError(f"Required node {node} not found")
                
        logger.info("Environment setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during environment setup: {str(e)}")
        raise

def process_latentsync(video_path: str, audio_path: str, output_path: str, 
                      width: int, height: int):
    """LatentSync 실행"""
    logger.info("Starting LatentSync processing...")
    
    try:
        from nodes import NODE_CLASS_MAPPINGS
        
        with torch.inference_mode():
            # LoadAudio
            loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
            loadaudio_output = loadaudio.load(audio=audio_path)

            # LoadVideo
            vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
            video_output = vhs_loadvideo.load_video(
                video=video_path,
                force_rate=25,
                custom_width=width,
                custom_height=height,
                frame_load_cap=0,
                skip_first_frames=0,
                select_every_nth=1,
                format="AnimateDiff",
                unique_id=random.randint(1, 2**32 - 1)
            )

            # VideoLengthAdjuster
            adjuster = NODE_CLASS_MAPPINGS["D_VideoLengthAdjuster"]()
            adjusted = adjuster.adjust(
                mode="pingpong",
                fps=25,
                silent_padding_sec=0.5,
                images=get_value_at_index(video_output, 0),
                audio=get_value_at_index(loadaudio_output, 0)
            )

            # LatentSync
            latentsync = NODE_CLASS_MAPPINGS["D_LatentSyncNode"]()
            synced = latentsync.inference(
                seed=random.randint(1, 2**32 - 1),
                images=get_value_at_index(adjusted, 0),
                audio=get_value_at_index(adjusted, 1)
            )

            # VideoCombine
            videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            result = videocombine.combine_video(
                frame_rate=25,
                loop_count=0,
                filename_prefix=os.path.splitext(os.path.basename(output_path))[0],
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(synced, 0),
                audio=get_value_at_index(synced, 1),
                unique_id=random.randint(1, 2**32 - 1)
            )

            logger.info("Processing completed successfully")
            return True

    except Exception as e:
        logger.error(f"Error during LatentSync processing: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run LatentSync")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--width", type=int, default=0, help="Custom width")
    parser.add_argument("--height", type=int, default=0, help="Custom height")
    
    args = parser.parse_args()
    
    logger.info("main 시작")
    logger.info(f"Input video: {args.video}")
    logger.info(f"Input audio: {args.audio}")
    logger.info(f"Output path: {args.output}")

    logger.info("comfyui_path 설정 전.........")
    comfyui_path = "/workspace/ComfyUI"
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    logger.info("comfyui_path 설정 후.........")

    # ComfyUI 모듈 import
    from ComfyUI import execution
    from ComfyUI import server
    from ComfyUI.nodes import NODE_CLASS_MAPPINGS, init_extra_nodes

    logger.info(f"PYTHONPATH : {os.environ.get('PYTHONPATH')}")
    logger.info(f"sys.path: {sys.path}")
    
    try:
        logger.info("setup_environment 전")
        setup_environment()
        logger.info("setup_environment 후")
        success = process_latentsync(args.video, args.audio, args.output, 
                                   args.width, args.height)
        
        if success:
            logger.info(f"LatentSync completed. Output saved to: {args.output}")
        else:
            logger.error("LatentSync failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()