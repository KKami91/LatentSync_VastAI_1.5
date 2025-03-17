# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import subprocess

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import AutoProcessor, Wav2Vec2Model

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_processor: AutoProcessor,
        audio_encoder: Wav2Vec2Model,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_processor=audio_processor,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.latent_space = vae is not None
        if self.latent_space:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        if self.latent_space:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            decoded_latents = self.vae.decode(latents).sample
        else:
            decoded_latents = rearrange(latents, "b c f h w -> (b f) c h w")
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        if self.latent_space:
            shape = (
                batch_size,
                num_channels_latents,
                1,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
        else:
            shape = (
                batch_size,
                num_channels_latents,
                1,
                height,
                width,
            )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        if self.latent_space:
            # resize the mask to latents shape as we concatenate the mask to the latents
            # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
            # and half precision
            mask = torch.nn.functional.interpolate(
                mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
            )
            masked_image = masked_image.to(device=device, dtype=dtype)

            # encode the mask image into latents space so we can concatenate it to the latents
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
            masked_image_latents = (
                masked_image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        else:
            masked_image_latents = masked_image.to(device=device, dtype=dtype)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        if self.latent_space:
            image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
            image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            image_latents = images
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def recover_original_pixel_values(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Combine the pixel values
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def crop_audio_window(self, original_mel, start_index):
        start_idx = int(80.0 * (start_index / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        return original_mel[:, start_idx:end_idx].unsqueeze(0)

    def affine_transform_video(self, video_path):
        """비디오 프레임에서 얼굴을 감지하고 변환하는 함수
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            tuple: 감지된 얼굴, 원본 비디오 프레임, 얼굴 위치 박스, 변환 행렬, 얼굴이 감지된 프레임 인덱스
            
        Raises:
            ValueError: 비디오의 모든 프레임에서 얼굴이 감지되지 않은 경우
        """
        # 비디오 프레임 읽기
        video_frames = read_video(video_path, use_decord=False)
        faces = []
        boxes = []
        affine_matrices = []
        frame_indices = []  # 얼굴이 감지된 프레임의 인덱스를 저장
        
        print(f"전체 {len(video_frames)}개 프레임에 대해 얼굴 변환 처리 중...")
        
        for i, frame in enumerate(tqdm.tqdm(video_frames)):
            try:
                # 프레임에서 얼굴 감지 시도
                face, box, affine_matrix = self.image_processor.affine_transform(frame)
                
                # 얼굴이 올바르게 감지되었는지 명시적으로 확인
                if face is not None and isinstance(face, torch.Tensor):
                    faces.append(face)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    frame_indices.append(i)  # 얼굴이 감지된 프레임 인덱스 저장
            except Exception as e:
                print(f"프레임 {i}에서 예외 발생: {str(e)}")
                if "No face detected" in str(e):
                    # 얼굴이 없는 프레임은 건너뜀
                    continue
                else:
                    # 다른 오류는 그대로 발생시킴
                    raise e
        
        if not faces:
            # 모든 프레임에서 얼굴이 감지되지 않은 경우
            raise ValueError("비디오의 모든 프레임에서 얼굴이 감지되지 않았습니다. 얼굴이 보이는 비디오를 사용해주세요.")
        
        # 감지된 얼굴만 처리
        faces = torch.stack(faces)
        
        # 얼굴이 감지된 프레임, 관련 메타데이터와 함께 반환
        return faces, video_frames, boxes, affine_matrices, frame_indices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """입력 비디오와 오디오를 사용하여 립싱크 처리를 수행합니다.
        
        Args:
            video_path: 입력 비디오 파일 경로
            audio_path: 입력 오디오 파일 경로
            video_out_path: 출력 비디오가 저장될 경로
            num_frames: 한 번에 처리할 프레임 수
            video_fps: 비디오 프레임 레이트
            기타 매개변수: 모델 추론 관련 설정들
            
        Returns:
            None: 결과는 video_out_path에 저장됩니다.
        """
        # 모델 평가 모드 설정
        is_train = self.unet.training
        self.unet.eval()

        # 0. 기본 매개변수 정의
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"샘플 프레임: {num_frames}")

        # 얼굴 감지 및 변환 수행
        video_frames, original_video_frames, boxes, affine_matrices, frame_indices = self.affine_transform_video(video_path)
        audio_samples = read_audio(audio_path)

        # 1. UNet에 기본 높이와 너비 설정
        if self.latent_space:
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. 입력 검증
        self.check_inputs(height, width, callback_steps)

        # 가이드 스케일 설정 (Imagen 논문의 식 (2)의 가이드 가중치 'w'와 유사하게 정의)
        # guidance_scale = 1은 classifier free guidance를 사용하지 않는 것을 의미
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. 타임스텝 설정
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. 추가 스텝 매개변수 준비
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        # 오디오 관련 설정
        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        # 얼굴이 감지된 프레임들을 num_frames 크기의 청크로 나누기
        frame_chunks = []
        for i in range(0, len(frame_indices), num_frames):
            chunk = frame_indices[i:i+num_frames]
            if len(chunk) == num_frames:  # 완전한 청크만 처리
                frame_chunks.append(chunk)

        num_inferences = len(frame_chunks)
        print(f"얼굴이 감지된 {num_inferences}개의 프레임 청크 처리 중")

        # 최종 결과 비디오를 위한 프레임 배열 초기화 (원본 비디오 프레임으로)
        final_frames = np.array(original_video_frames)
        
        # 잠재 변수 준비
        if self.latent_space:
            num_channels_latents = self.vae.config.latent_channels
        else:
            num_channels_latents = 3

        # 각 청크 처리
        for chunk_idx, chunk in enumerate(tqdm.tqdm(frame_chunks, desc="얼굴 청크 처리 중")):
            # 현재 청크의 프레임들 가져오기
            chunk_frames = [video_frames[frame_indices.index(idx)] for idx in chunk]
            chunk_frames_tensor = torch.stack(chunk_frames)
            
            # 현재 청크의 박스와 변환 행렬 가져오기
            chunk_boxes = [boxes[frame_indices.index(idx)] for idx in chunk]
            chunk_affine_matrices = [affine_matrices[frame_indices.index(idx)] for idx in chunk]
            
            # 현재 청크에 대한 latent 준비
            latents = self.prepare_latents(
                batch_size,
                num_frames,
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )

            # 오디오 특성 준비
            if self.unet.add_audio_layer:
                # 현재 청크에 해당하는 오디오 특성 가져오기
                mel_overlap = torch.stack([whisper_chunks[idx] for idx in chunk])
                mel_overlap = mel_overlap.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    empty_mel_overlap = torch.zeros_like(mel_overlap)
                    mel_overlap = torch.cat([empty_mel_overlap, mel_overlap])
            else:
                mel_overlap = None

            # 마스크 및 마스크된 이미지 준비
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                chunk_frames_tensor, affine_transform=False
            )

            # 7. 마스크 잠재 변수 준비
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. 이미지 잠재 변수 준비
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. 디노이징 루프
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # classifier free guidance를 사용하는 경우 latent 확장
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # latent, 마스크, 마스크된 이미지 latent를 채널 차원에서 결합
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # 노이즈 잔차 예측
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=mel_overlap).sample

                    # 이전의 노이즈 샘플 계산 x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # 콜백 호출 (제공된 경우)
                    if j == len(timesteps) - 1 or (
                        (j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # 픽셀 값 복원
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.recover_original_pixel_values(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            
            # 처리된 얼굴 프레임 획득
            processed_faces = decoded_latents
            
            # 원본 비디오에 처리된 얼굴 적용
            for i, frame_idx in enumerate(chunk):
                try:
                    face = processed_faces[i]
                    x1, y1, x2, y2 = chunk_boxes[i]
                    height_box = int(y2 - y1)
                    width_box = int(x2 - x1)
                    
                    # 얼굴 크기 조정
                    face = torchvision.transforms.functional.resize(face, size=(height_box, width_box), antialias=True)
                    face = rearrange(face, "c h w -> h w c")
                    face = (face / 2 + 0.5).clamp(0, 1)
                    face = (face * 255).to(torch.uint8).cpu().numpy()
                    
                    # 원본 프레임에 처리된 얼굴 적용
                    restored_frame = self.image_processor.restorer.restore_img(
                        original_video_frames[frame_idx], 
                        face, 
                        chunk_affine_matrices[i]
                    )
                    
                    # 최종 결과 비디오 프레임 업데이트
                    final_frames[frame_idx] = restored_frame
                except Exception as e:
                    print(f"프레임 {frame_idx} 처리 중 오류 발생: {str(e)}")
                    # 오류 발생 시 원본 프레임 유지
                    continue

        # 오디오 길이에 맞게 비디오 길이 조정
        audio_samples_remain_length = int(len(final_frames) / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        # 훈련 모드 복원
        if is_train:
            self.unet.train()

        # 임시 디렉토리 준비
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # 최종 비디오 저장
        write_video(os.path.join(temp_dir, "video.mp4"), final_frames, fps=25)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        # 비디오와 오디오 결합
        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
