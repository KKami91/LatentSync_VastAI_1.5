import os
import math
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import sys
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
import shutil
import decimal
from decimal import Decimal, ROUND_UP
import requests

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers'
    ]

    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None

    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")

    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installing required package: {package}")
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def download_model(url, save_path):
    """Download a model from a URL and save it to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def pre_download_models():
    """Pre-download all required models."""
    models = {
        "s3fd-e19a316812.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-e19a316812.pth",
        # 불필요한 모델 제거
    }

    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    for model_name, url in models.items():
        save_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(save_path):
            print(f"Downloading {model_name}...")
            download_model(url, save_path)
        else:
            print(f"{model_name} already exists in cache.")

def setup_models():
    """Setup and pre-download all required models."""
    # Pre-download additional models
    pre_download_models()

    # Existing setup logic for LatentSync models
    cur_dir = get_ext_dir()
    ckpt_dir = os.path.join(cur_dir, "checkpoints")
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    if not (os.path.exists(unet_path) and os.path.exists(whisper_path)):
        print("Downloading required model checkpoints... This may take a while.")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync",
                             allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                             local_dir=ckpt_dir, local_dir_use_symlinks=False)
            print("Model checkpoints downloaded successfully!")
        except Exception as e:
            print(f"Error downloading models: {str(e)}")
            print("\nPlease download models manually:")
            print("1. Visit: https://huggingface.co/chunyu-li/LatentSync")
            print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
            print(f"3. Place them in: {ckpt_dir}")
            print(f"   with whisper/tiny.pt in: {whisper_dir}")
            raise RuntimeError("Model download failed. See instructions above.")

class LatentSyncNode:
    def __init__(self):
        check_and_install_dependencies()
        setup_models()  # This will now pre-download all required models

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                 },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def inference(self, images, audio, seed):
        # Existing inference logic
        torch.cuda.empty_cache()
        cur_dir = get_ext_dir()
        ckpt_dir = os.path.join(cur_dir, "checkpoints")
        output_dir = folder_paths.get_output_directory()
        temp_dir = os.path.join(output_dir, "temp_frames")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Create a temporary video file from the input frames
        output_name = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_video_path = os.path.join(output_dir, f"temp_{output_name}.mp4")
        output_video_path = os.path.join(output_dir, f"latentsync_{output_name}_out.mp4")

        # Save frames as temporary video
        import torchvision.io as io
        if isinstance(images, list):
            frames = torch.stack(images)
        else:
            frames = images
        print(f"Initial frame count: {frames.shape[0]}")

        frames = (frames * 255).byte()
        if len(frames.shape) == 3:
            frames = frames.unsqueeze(0)
        print(f"Frame count before writing video: {frames.shape[0]}")

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu()
        try:
            io.write_video(temp_video_path, frames, fps=25, video_codec='h264')
        except TypeError:
            # Fallback for newer versions
            import av
            container = av.open(temp_video_path, mode='w')
            stream = container.add_stream('h264', rate=25)
            stream.width = frames.shape[2]
            stream.height = frames.shape[1]
            
            for frame in frames:
                frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)
            
            # Flush stream
            packet = stream.encode(None)
            container.mux(packet)
            container.close()
        video_path = normalize_path(temp_video_path)
        torch.cuda.empty_cache()

        if not os.path.exists(ckpt_dir):
            print("Downloading model checkpoints... This may take a while.")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync",
                                    allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                                    local_dir=ckpt_dir, local_dir_use_symlinks=False)
            print("Model checkpoints downloaded successfully!")

        inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
        unet_config_path = normalize_path(os.path.join(cur_dir, "configs", "unet", "second_stage.yaml"))
        scheduler_config_path = normalize_path(os.path.join(cur_dir, "configs"))
        ckpt_path = normalize_path(os.path.join(ckpt_dir, "latentsync_unet.pt"))
        whisper_ckpt_path = normalize_path(os.path.join(ckpt_dir, "whisper", "tiny.pt"))

        # resample audio to 16k hz and save to wav
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.dim() == 3: # Expected shape: [channels, samples]
            waveform = waveform.squeeze(0)

        if sample_rate != 16000:
            new_sample_rate = 16000
            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)
            waveform, sample_rate = waveform_16k, new_sample_rate

        # Package resampled audio
        resampled_audio = {
            "waveform": waveform.unsqueeze(0),  # Add batch dim
            "sample_rate": sample_rate
        }
        torch.cuda.empty_cache()

        audio_path = normalize_path(os.path.join(output_dir, f"latentsync_{output_name}_audio.wav"))
        torchaudio.save(audio_path, waveform, sample_rate)

        print(f"Using video path: {video_path}")
        print(f"Video file exists: {os.path.exists(video_path)}")
        print(f"Video file size: {os.path.getsize(video_path)} bytes")

        assert os.path.exists(video_path), f"video_path not exists: {video_path}"
        assert os.path.exists(audio_path), f"audio_path not exists: {audio_path}"

        try:
            torch.cuda.empty_cache()
            # Add the package root to Python path
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            
            # Add the current directory to Python path
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
        
            # Create a Namespace object with the arguments
            args = argparse.Namespace(
                unet_config_path=unet_config_path,
                inference_ckpt_path=ckpt_path,
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path
            )
        
            # Load the config
            config = OmegaConf.load(unet_config_path)
        
            try:
                # Call main with both config and args
                inference_module.main(config, args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                # 향상된 오류 처리
                error_msg = str(e)
                
                # 얼굴 감지 관련 오류 케이스
                face_detection_errors = [
                    "No face detected in any frame",
                    "얼굴 감지 실패",
                    "비디오의 모든 프레임에서 얼굴이 감지되지 않았습니다",
                    "expected Tensor as element"
                ]
                
                if any(err in error_msg for err in face_detection_errors):
                    print("얼굴 감지 실패: 비디오에서 충분한 얼굴이 감지되지 않았습니다.")
                    # 이 오류는 API 핸들러에서 처리할 수 있도록 전파
                    raise RuntimeError("얼굴 감지 실패: 영상에서 얼굴을 찾을 수 없거나 일부 프레임에만 얼굴이 있습니다.")
                else:
                    # 기타 예상치 못한 오류
                    print(f"예상치 못한 오류 발생: {error_msg}")
                    raise e

            # Load the processed video back as frames
            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]  # [T, H, W, C]
            print(f"Frame count after reading video: {processed_frames.shape[0]}")
            torch.cuda.empty_cache()
            # Process frames following wav2lip.py pattern
            out_tensor_list = []
            for frame in processed_frames:
                # Convert to numpy and ensure correct format
                frame = frame.numpy()
                
                # Convert frame to float32 and normalize
                frame = frame.astype(np.float32) / 255.0
                
                # Convert back to tensor
                frame = torch.from_numpy(frame)
                
                # Ensure we have 3 channels
                if len(frame.shape) == 2:  # If grayscale
                    frame = frame.unsqueeze(2).repeat(1, 1, 3)
                elif frame.shape[2] == 4:  # If RGBA
                    frame = frame[:, :, :3]
                
                # Change to [C, H, W] format
                frame = frame.permute(2, 0, 1)
                
                out_tensor_list.append(frame)

            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]  # [T, H, W, C]
            processed_frames = processed_frames.float() / 255.0
            print(f"Frame count after normalization: {processed_frames.shape[0]}")
            torch.cuda.empty_cache()
            # Fix dimensions for VideoCombine compatibility
            if len(processed_frames.shape) == 3:  
                processed_frames = processed_frames.unsqueeze(0)
            if processed_frames.shape[0] == 1 and len(processed_frames.shape) == 4:
                processed_frames = processed_frames.squeeze(0)
            if processed_frames.shape[0] == 3:  # If in CHW format
                processed_frames = processed_frames.permute(1, 2, 0)  # Convert to HWC
            if processed_frames.shape[-1] == 4:  # If RGBA
                processed_frames = processed_frames[..., :3]

            print(f"Final frame count: {processed_frames.shape[0]}")
            print(f"Final shape: {processed_frames.shape}")

            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # For specific errors, propagate to the API handler for fallback handling
            if "No face detected" in str(e) or "얼굴 감지 실패" in str(e):
                raise RuntimeError(f"Face detection issue: {str(e)}")
            raise

        return (processed_frames, resampled_audio)

class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
            # Bypass video frames exactly
            video_duration = len(original_frames) / fps
            required_samples = int(video_duration * sample_rate)
            
            # Adjust audio to match video duration
            if waveform.shape[1] >= required_samples:
                adjusted_audio = waveform[:, :required_samples]  # Trim audio
            else:
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)  # Pad audio
            
            return (
                torch.stack(original_frames),
                {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            # Add silent padding then pingpong loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
            frames = original_frames + reversed_frames
            while len(frames) < target_frames:
                frames += frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "loop_to_audio":
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

NODE_CLASS_MAPPINGS = {
    "D_LatentSyncNode": LatentSyncNode,
    "D_VideoLengthAdjuster": VideoLengthAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_LatentSyncNode": "LatentSync Node",
    "D_VideoLengthAdjuster": "Video Length Adjuster",
}

