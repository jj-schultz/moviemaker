#!/usr/bin/env python3
import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
from moviepy import ImageSequenceClip, AudioFileClip, VideoFileClip, ImageClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BeatGrid:
    """Container for beat/anchor metadata derived from the audio track."""
    
    tempo_bpm: float
    beat_times: List[float]
    anchor_times: List[float]
    beats_per_measure: int
    measures_per_long: int
    source: str = "detected"
    
    @property
    def anchor_interval(self) -> Optional[float]:
        if self.tempo_bpm <= 0:
            return None
        beats_per_anchor = self.beats_per_measure * self.measures_per_long
        seconds_per_beat = 60.0 / self.tempo_bpm
        return beats_per_anchor * seconds_per_beat


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _dedupe_times(times: List[float], epsilon: float = 1e-3) -> List[float]:
    deduped: List[float] = []
    for moment in times:
        if deduped and abs(moment - deduped[-1]) < epsilon:
            continue
        deduped.append(moment)
    return deduped


def _extend_anchor_series(
        anchors: List[float],
        tempo_bpm: float,
        audio_duration: float,
        beats_per_measure: int,
        measures_per_long: int,
) -> List[float]:
    if audio_duration <= 0:
        return anchors
    
    beats_per_anchor = beats_per_measure * measures_per_long
    if beats_per_anchor <= 0:
        return anchors
    
    seconds_per_anchor: Optional[float] = None
    if tempo_bpm > 0:
        seconds_per_anchor = (60.0 / tempo_bpm) * beats_per_anchor
    
    if seconds_per_anchor is None or seconds_per_anchor <= 0:
        # Fallback to evenly spreading anchors over the full duration
        seconds_per_anchor = audio_duration / max(1, len(anchors) or beats_per_anchor)
    
    extended = list(anchors)
    if not extended:
        extended.append(0.0)
    
    last_anchor = extended[-1]
    while last_anchor + seconds_per_anchor <= audio_duration + 1e-3:
        last_anchor += seconds_per_anchor
        extended.append(last_anchor)
    
    return extended


def _validate_image_path(image_path: Optional[str], label: str) -> Optional[str]:
    if not image_path:
        return None
    normalized = os.path.abspath(image_path)
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"{label} image path not found: {image_path}")
    return normalized


def analyze_audio_beats(
        audio_path: str,
        measures_per_long: int = 4,
        beats_per_measure: int = 4,
        beat_detection_sensitivity: float = 0.6,
) -> Optional[BeatGrid]:
    """Extract beat and measure anchors using librosa onset/beat detection."""
    if measures_per_long <= 0 or beats_per_measure <= 0:
        logger.warning(
            "Invalid measure configuration (measures_per_long=%s, beats_per_measure=%s); "
            "skipping beat alignment",
            measures_per_long,
            beats_per_measure,
        )
        return None
    
    try:
        import librosa
    except ImportError:
        logger.warning("librosa is not installed; beat alignment disabled")
        return None
    
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        audio_duration = float(librosa.get_duration(y=y, sr=sr))
    except Exception:
        logger.exception("Failed to load audio for beat analysis")
        return None
    
    hop_length = 512
    sensitivity = _clamp(beat_detection_sensitivity, 0.05, 0.95)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tightness = _clamp(100 + (0.5 - sensitivity) * 200, 10.0, 500.0)
    
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            tightness=tightness,
        )
    except Exception:
        logger.exception("librosa.beat.beat_track failed; beat alignment disabled")
        return None
    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()
    if not beat_times:
        delta = 0.2 + (1.0 - sensitivity) * 0.3
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            delta=delta,
        )
        beat_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length).tolist()
    
    beat_times = _dedupe_times([t for t in beat_times if t >= 0.0])
    
    if (tempo is None or tempo <= 0.0) and len(beat_times) >= 2:
        avg_spacing = float(np.median(np.diff(beat_times)))
        if avg_spacing > 0:
            tempo = 60.0 / avg_spacing
    
    if tempo is None or tempo <= 0.0:
        tempo = 120.0
    
    beats_per_anchor = beats_per_measure * measures_per_long
    anchor_times: List[float] = []
    if beat_times:
        anchor_times = [beat_times[i] for i in range(0, len(beat_times), beats_per_anchor)]
    anchor_times = _dedupe_times(anchor_times)
    anchor_times = _extend_anchor_series(
        anchor_times,
        tempo,
        audio_duration,
        beats_per_measure,
        measures_per_long,
    )
    
    logger.info(
        "Beat grid prepared: tempo=%.2f BPM, beats=%d, anchors=%d (measures_per_long=%d)",
        tempo,
        len(beat_times),
        len(anchor_times),
        measures_per_long,
    )
    
    return BeatGrid(
        tempo_bpm=tempo,
        beat_times=beat_times,
        anchor_times=anchor_times,
        beats_per_measure=beats_per_measure,
        measures_per_long=measures_per_long,
        source="detected" if beat_times else "synthetic",
    )


def derive_anchor_times(
        beat_grid: Optional[BeatGrid],
        start_time: float,
        end_time: float,
        long_duration: float,
        anchor_delay: float = 0.0,
) -> List[float]:
    if not beat_grid or not beat_grid.anchor_times:
        return []
    
    usable: List[float] = []
    window_start = max(0.0, start_time)
    window_end = max(window_start, end_time - long_duration - max(0.0, anchor_delay))
    for anchor in beat_grid.anchor_times:
        if anchor < window_start - 1e-3:
            continue
        if anchor > window_end + 1e-3:
            break
        if usable and abs(anchor - usable[-1]) < 1e-3:
            continue
        usable.append(anchor)
    
    return usable


def get_image_files(images_dir):
    """Get all PNG/JPG image files and video files sorted chronologically"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    video_extensions = ['*.mov', '*.mp4', '*.avi', '*.mkv']
    media_files = []
    
    # Get image files
    for extension in image_extensions:
        pattern = os.path.join(images_dir, extension)
        media_files.extend(glob.glob(pattern))
    
    # Get video files
    for extension in video_extensions:
        pattern = os.path.join(images_dir, extension)
        media_files.extend(glob.glob(pattern))
    
    # Sort by modification time (chronological order)
    media_files.sort(key=os.path.getmtime)
    image_count = sum(1 for f in media_files if any(f.lower().endswith(ext[1:]) for ext in image_extensions))
    video_count = len(media_files) - image_count
    logger.info(f"Found {len(media_files)} media files in {images_dir} ({image_count} images, {video_count} videos)")
    return media_files


def get_special_images(images_long, first_override=None, last_override=None):
    """Get special 'first' and 'last' images, honoring optional overrides"""
    first_img = first_override
    last_img = last_override
    regular_long = []

    first_override_norm = os.path.abspath(first_override) if first_override else None
    last_override_norm = os.path.abspath(last_override) if last_override else None

    for img_path in images_long:
        normalized = os.path.abspath(img_path)
        if first_override_norm and normalized == first_override_norm:
            continue
        if last_override_norm and normalized == last_override_norm:
            continue

        filename = os.path.splitext(os.path.basename(img_path))[0].lower()
        if not first_img and filename == 'first':
            first_img = img_path
        elif not last_img and filename == 'last':
            last_img = img_path
        else:
            regular_long.append(img_path)

    return first_img, last_img, regular_long


def calculate_progressive_duration(progress_ratio, short_start_duration=0.75,
                                   short_end_duration=0.20, acceleration_rate=2.0):
    """Calculate duration for short image based on progress through movie
    Args:
        progress_ratio: Float between 0.0 (start) and 1.0 (end)
        short_start_duration: Duration at start in seconds (default 0.75)
        short_end_duration: Duration at end in seconds (default 0.20)
        acceleration_rate: Controls how quickly durations shrink (>= 1.0)
    Returns:
        Duration in seconds using an accelerated easing curve
    """
    if acceleration_rate < 1.0:
        raise ValueError("acceleration_rate must be >= 1.0")
    
    clamped_progress = max(0.0, min(1.0, progress_ratio))
    eased_progress = clamped_progress ** (1.0 / acceleration_rate)
    return short_start_duration - (short_start_duration - short_end_duration) * eased_progress


def create_movie_sequence(images_short, images_long, audio_duration,
                          long_duration=3.0, short_start_duration=0.75, short_end_duration=0.20,
                          short_acceleration=2.0, first_image_duration=2.0, last_image_duration=2.0,
                          beat_grid: Optional[BeatGrid] = None, random_seed: Optional[int] = None,
                          long_anchor_delay: float = 0.05,
                          first_image_override: Optional[str] = None,
                          last_image_override: Optional[str] = None):
    """Create a sequence with evenly spaced long images and progressive short timing
    Args:
        images_short: List of short image file paths
        images_long: List of long image file paths
        audio_duration: Total duration in seconds
        long_duration: Duration for long images in seconds (default 3.0)
        short_start_duration: Starting duration for short images in seconds (default 0.75)
        short_end_duration: Ending duration for short images in seconds (default 0.20)
        short_acceleration: Rate controlling how quickly short durations shrink (default 2.0)
        first_image_duration: Duration for the optional "first" image (default 2.0)
        last_image_duration: Duration for the optional "last" image (default 2.0)
        beat_grid: Optional BeatGrid metadata to align long images to detected beats
        random_seed: Optional seed for deterministic shuffling and overflow sampling
        long_anchor_delay: Seconds to delay long-image onset after anchor beats (default 0.05s)
        first_image_override: Optional absolute path to force as first image
        last_image_override: Optional absolute path to force as last image
    """
    import random
    
    rng = random.Random(random_seed) if random_seed is not None else random
    
    if long_duration <= 0:
        raise ValueError("long_duration must be greater than 0")
    if first_image_duration < 0 or last_image_duration < 0:
        raise ValueError("first_image_duration and last_image_duration must be >= 0")
    
    first_img, last_img, regular_long = get_special_images(
        images_long,
        first_override=first_image_override,
        last_override=last_image_override,
    )
    short_pool = images_short.copy()
    
    if regular_long:
        rng.shuffle(regular_long)
    final_sequence = []  # List of (image_path, duration, type)
    
    if first_img:
        final_sequence.append((first_img, first_image_duration, 'first'))
        logger.info("Added 'first' image at start for %.2f seconds", first_image_duration)
    
    start_time = first_image_duration if first_img else 0.0
    end_time = (audio_duration - last_image_duration) if last_img else audio_duration
    if end_time < start_time:
        raise ValueError("Audio duration is too short for the requested first/last images")
    
    available_duration = end_time - start_time
    logger.info(
        "Available duration for main content: %.2fs (from %.1fs to %.1fs)",
        available_duration,
        start_time,
        end_time,
    )
    
    anchor_times = (
        derive_anchor_times(
            beat_grid,
            start_time,
            end_time,
            long_duration,
            anchor_delay=max(0.0, long_anchor_delay),
        )
        if beat_grid
        else []
    )
    beat_alignment_active = bool(anchor_times)
    if beat_alignment_active:
        logger.info(
            "Beat alignment active: %d anchor slots between %.2fs and %.2fs (tempo=%.2f BPM, source=%s)",
            len(anchor_times),
            start_time,
            end_time,
            beat_grid.tempo_bpm if beat_grid else 0.0,
            beat_grid.source if beat_grid else "n/a",
        )
    elif beat_grid:
        logger.warning(
            "Beat grid provided but no anchors fell inside %.2fs window; reverting to spacing",
            available_duration,
        )
        beat_grid = None
    
    gap_hint = max(short_start_duration, short_end_duration, 0.1)
    max_long_by_gap = int(available_duration / (long_duration + gap_hint)) if available_duration > 0 else 0
    max_long_by_duration = int(available_duration / long_duration) if available_duration > 0 else 0
    
    if beat_alignment_active:
        max_long_slots = min(len(anchor_times), max_long_by_duration)
    else:
        max_long_slots = min(max_long_by_gap, max_long_by_duration)
    
    target_long_count = max(0, max_long_slots)
    
    if len(regular_long) > target_long_count:
        overflow_count = len(regular_long) - target_long_count
        if overflow_count > 0:
            overflow_set = set(rng.sample(regular_long, overflow_count))
            short_pool.extend(list(overflow_set))
            regular_long = [img for img in regular_long if img not in overflow_set]
            logger.info(
                "Reclassified %d long images into short pool to keep spacing consistent",
                overflow_count,
            )
    elif len(regular_long) < target_long_count:
        deficit = target_long_count - len(regular_long)
        promotable = min(deficit, len(short_pool)) if short_pool else 0
        if promotable > 0:
            promoted = rng.sample(short_pool, promotable)
            promoted_set = set(promoted)
            short_pool = [img for img in short_pool if img not in promoted_set]
            regular_long.extend(promoted)
            logger.info(
                "Promoted %d short images into long pool to satisfy anchor spacing",
                promotable,
            )
        if len(regular_long) < target_long_count:
            logger.warning(
                "Needed %d long images but only %d available even after promotion",
                target_long_count,
                len(regular_long),
            )
            target_long_count = len(regular_long)
    else:
        logger.info("All %d candidate long images retained", len(regular_long))
    
    long_images = regular_long[:target_long_count]
    long_total_duration = len(long_images) * long_duration
    available_short_time = max(0.0, available_duration - long_total_duration)
    logger.info(
        "Scheduling %d long images (%.2fs total) with %.2fs short-image budget",
        len(long_images),
        long_total_duration,
        available_short_time,
    )
    
    anchor_schedule: List[float] = []
    if beat_alignment_active:
        anchor_schedule = anchor_times[:len(long_images)]
        if len(anchor_schedule) < len(anchor_times):
            logger.info(
                "Using %d of %d available anchors based on remaining long images",
                len(anchor_schedule),
                len(anchor_times),
            )
        if not anchor_schedule:
            beat_alignment_active = False
    
    if beat_alignment_active and not short_pool:
        logger.warning("Beat alignment requires at least one short image; falling back to spacing mode")
        beat_alignment_active = False
        anchor_schedule = []
    
    gap_count = len(long_images) + 1 if available_duration > 0 else 0
    avg_short_duration = max(0.05, (short_start_duration + short_end_duration) / 2)
    estimated_short_count = max(1, int(available_short_time / avg_short_duration) + 10)
    
    short_durations = []
    for i in range(estimated_short_count):
        progress = i / max(1, estimated_short_count - 1)
        duration = calculate_progressive_duration(
            progress,
            short_start_duration,
            short_end_duration,
            short_acceleration,
        )
        short_durations.append(duration)
    
    short_duration_index = 0
    short_img_index = 0
    if short_pool:
        rng.shuffle(short_pool)
    
    def draw_short_image():
        nonlocal short_img_index
        if not short_pool:
            return None
        if short_img_index >= len(short_pool):
            random.shuffle(short_pool)
            short_img_index = 0
        image_path = short_pool[short_img_index]
        short_img_index += 1
        return image_path
    
    def peek_short_duration():
        if short_duration_index < len(short_durations):
            return short_durations[short_duration_index]
        return short_end_duration
    
    def consume_short_duration():
        nonlocal short_duration_index
        if short_duration_index < len(short_durations):
            duration = short_durations[short_duration_index]
            short_duration_index += 1
            return duration
        short_duration_index += 1
        return short_end_duration
    
    remaining_short_time = available_short_time
    
    def fill_gap(target_duration, pad_with_leftover=False):
        nonlocal remaining_short_time
        if target_duration <= 0 or not short_pool:
            return 0.0
        
        gap_entries: List[tuple[str, float]] = []
        planned_total = 0.0
        
        while True:
            duration = peek_short_duration()
            if duration <= 0:
                break
            if planned_total > 0 and planned_total + duration > target_duration + 1e-9:
                break
            image_path = draw_short_image()
            if image_path is None:
                break
            duration = consume_short_duration()
            gap_entries.append((image_path, duration))
            planned_total += duration
            if planned_total >= target_duration - 1e-6:
                break
        
        if not gap_entries:
            image_path = draw_short_image()
            if image_path is None:
                return 0.0
            duration = consume_short_duration()
            if duration <= 0:
                duration = short_end_duration
            gap_entries.append((image_path, max(duration, 1e-3)))
            planned_total = gap_entries[0][1]
        
        num_entries = len(gap_entries)
        if num_entries == 0:
            return 0.0
        
        desired_total = target_duration if target_duration > 0 else planned_total
        if desired_total <= 0:
            desired_total = planned_total
        if desired_total <= 0:
            return 0.0
        
        # Evenly space short images inside this gap by using a uniform duration that
        # matches the average time the acceleration curve scheduled for this slot.
        uniform_duration = desired_total / num_entries
        filled = 0.0
        for image_path, _ in gap_entries:
            final_sequence.append((image_path, uniform_duration, 'short'))
            filled += uniform_duration
        
        remaining_short_time = max(0.0, remaining_short_time - filled)
        return filled
    
    if beat_alignment_active and anchor_schedule:
        current_time = start_time
        anchor_delay = max(0.0, long_anchor_delay)
        for idx, anchor_time in enumerate(anchor_schedule):
            desired_start = anchor_time + anchor_delay
            desired_start = min(desired_start, end_time - long_duration)
            desired_start = max(desired_start, current_time)
            gap_duration = max(0.0, desired_start - current_time)
            if gap_duration > 1e-6:
                filled = fill_gap(gap_duration, pad_with_leftover=True)
                if abs(filled - gap_duration) > 1e-3:
                    logger.warning(
                        "Gap fill mismatch before beat anchor: expected %.3fs, filled %.3fs",
                        gap_duration,
                        filled,
                    )
                current_time += filled
            
            if idx < len(long_images):
                final_sequence.append((long_images[idx], long_duration, 'long'))
                current_time += long_duration
        
        tail_gap = max(0.0, end_time - current_time)
        if tail_gap > 1e-6:
            filled = fill_gap(tail_gap, pad_with_leftover=True)
            if abs(filled - tail_gap) > 1e-3:
                logger.warning(
                    "Tail gap fill mismatch: expected %.3fs, filled %.3fs",
                    tail_gap,
                    filled,
                )
        logger.info(
            "Anchored %d long images to beat grid (anchor interval=%.2fs)",
            len(anchor_schedule),
            beat_grid.anchor_interval if beat_grid else 0.0,
        )
    elif gap_count > 0:
        for gap_index in range(gap_count):
            remaining_gaps = gap_count - gap_index
            target_duration = (remaining_short_time / remaining_gaps) if remaining_gaps else 0.0
            fill_gap(target_duration)
            
            if gap_index < len(long_images):
                final_sequence.append((long_images[gap_index], long_duration, 'long'))
    
    if last_img:
        final_sequence.append((last_img, last_image_duration, 'last'))
        logger.info("Added 'last' image at end for %.2f seconds", last_image_duration)
    
    final_images = [item[0] for item in final_sequence]
    final_durations = [item[1] for item in final_sequence]
    
    short_count = sum(1 for item in final_sequence if item[2] == 'short')
    long_count = sum(1 for item in final_sequence if item[2] == 'long')
    
    logger.info("Created sequence: %d total images", len(final_images))
    logger.info(
        "- %d short images with progressive timing: %.3fs -> %.3fs",
        short_count,
        short_start_duration,
        short_end_duration,
    )
    logger.info("- %d long images with %.1fs duration each", long_count, long_duration)
    logger.info("Final duration: %.2fs, Target: %.2fs", sum(final_durations), audio_duration)
    
    return final_images, final_durations


def is_video_file(file_path):
    """Check if a file is a video file based on its extension"""
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


def process_video_clip(video_path, duration, target_size=(1920, 1080)):
    """Process video file by extracting the first 'duration' seconds and stripping audio"""
    import sys
    import contextlib
    from io import StringIO
    
    original_clip = None
    try:
        # Suppress "Proc not detected" output from MoviePy
        with contextlib.redirect_stdout(StringIO()):
            # Load video clip
            original_clip = VideoFileClip(video_path)
        
        # Extract 'duration' seconds from the middle of the video
        video_clip = original_clip
        if original_clip.duration > duration:
            # Calculate the start time to extract from the middle
            start_time = (original_clip.duration - duration) / 2
            end_time = start_time + duration
            video_clip = original_clip.subclipped(start_time, end_time)
            logger.debug(f"Extracted {duration:.2f}s from middle of {video_path} (t={start_time:.2f}s to {end_time:.2f}s)")
        
        # Strip audio
        video_clip = video_clip.without_audio()
        
        # Resize while maintaining aspect ratio (like resize_image_to_standard does)
        video_width, video_height = video_clip.size
        target_width, target_height = target_size
        
        # Calculate scaling to fit within target size while maintaining aspect ratio
        video_ratio = video_width / video_height
        target_ratio = target_width / target_height
        
        if video_ratio > target_ratio:
            # Video is wider, fit by width
            new_width = target_width
            new_height = int(new_width / video_ratio)
        else:
            # Video is taller, fit by height
            new_height = target_height
            new_width = int(new_height * video_ratio)
        
        # Resize the video to the calculated dimensions
        resized_clip = video_clip.resized((new_width, new_height))
        
        # Add black borders to center the video in the target frame size
        if new_width != target_width or new_height != target_height:
            # Create black background
            background = ColorClip(size=target_size, color=(0, 0, 0), duration=resized_clip.duration)
            
            # Calculate position to center the video
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Composite the video on the black background
            final_clip = CompositeVideoClip([
                background,
                resized_clip.with_position((x_offset, y_offset))
            ])
            
            # Close intermediate clips
            background.close()
            resized_clip.close()
        else:
            final_clip = resized_clip
        
        # Close the video_clip if it's different from original_clip (i.e., if it was subclipped)
        if video_clip != original_clip:
            video_clip.close()
            
        return final_clip
        
    except Exception as e:
        if original_clip:
            original_clip.close()
        logger.exception(f"Error processing video {video_path}")
        raise


def resize_image_to_standard(image_path, target_size=(1920, 1080)):
    """Resize image to standard video dimensions while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling to fit within target size while maintaining aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider, fit by width
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller, fit by height
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            
            # Resize the image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a black background of target size
            background = Image.new('RGB', target_size, (0, 0, 0))
            
            # Paste the resized image centered on the background
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            background.paste(img_resized, (x_offset, y_offset))
            
            return np.array(background)
    except Exception as e:
        logger.exception(f"Error processing image {image_path}")
        raise


def create_movie(
    images_short_dir,
    images_long_dir,
    audio_path,
    output_path,
    long_duration=3.0,
    short_start_duration=0.75,
    short_end_duration=0.20,
    short_acceleration=2.0,
    first_image_duration=2.0,
    last_image_duration=2.0,
    measures_per_long=4,
    beats_per_measure=4,
    beat_detection_sensitivity=0.6,
    enable_beat_grid=True,
    random_seed: Optional[int] = None,
    long_anchor_delay=0.05,
    first_image_path: Optional[str] = None,
    last_image_path: Optional[str] = None,
):
    """Create movie from images and audio using new directory structure
    Args:
        images_short_dir: Directory containing short images
        images_long_dir: Directory containing long images
        audio_path: Path to audio file
        output_path: Path for output video
        long_duration: Duration for long images in seconds (default 3.0)
        short_start_duration: Starting duration for short images in seconds (default 0.75)
        short_end_duration: Ending duration for short images in seconds (default 0.20)
        short_acceleration: Rate controlling how quickly short durations shrink (default 2.0)
        first_image_duration: Duration for the optional "first" image (default 2.0)
        last_image_duration: Duration for the optional "last" image (default 2.0)
        measures_per_long: Number of measures between long-image anchor points (default 4)
        beats_per_measure: Beats per measure for anchor math (default 4, assumes 4/4)
        beat_detection_sensitivity: Float 0-1 describing onset sensitivity (default 0.6)
        enable_beat_grid: Toggle beat alignment (defaults to True)
        random_seed: Optional seed passed to sequence builder for deterministic shuffling
        long_anchor_delay: Seconds to delay long-image onset relative to the downbeat (default 0.05s)
        first_image_path: Optional absolute path to force as the opening frame
        last_image_path: Optional absolute path to force as the closing frame
    """
    try:
        # Validate input files exist
        if not os.path.exists(images_short_dir):
            raise FileNotFoundError(f"Short images directory not found: {images_short_dir}")
        
        if not os.path.exists(images_long_dir):
            raise FileNotFoundError(f"Long images directory not found: {images_long_dir}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        first_override = _validate_image_path(first_image_path, "First")
        last_override = _validate_image_path(last_image_path, "Last")

        if first_override:
            logger.info("Using explicit first image: %s", first_override)
        if last_override:
            logger.info("Using explicit last image: %s", last_override)

        # Get image files from both directories
        images_short = get_image_files(images_short_dir)
        images_long = get_image_files(images_long_dir)

        if not images_short and not images_long and not (first_override or last_override):
            raise ValueError("No image files found in either directory")

        override_norms = {
            os.path.abspath(path)
            for path in (first_override, last_override)
            if path
        }

        if override_norms:
            images_short = [
                img for img in images_short if os.path.abspath(img) not in override_norms
            ]
            images_long = [
                img for img in images_long if os.path.abspath(img) not in override_norms
            ]
        beat_grid = None
        if enable_beat_grid:
            beat_grid = analyze_audio_beats(
                audio_path,
                measures_per_long=measures_per_long,
                beats_per_measure=beats_per_measure,
                beat_detection_sensitivity=beat_detection_sensitivity,
            )
        
        # Load audio to get duration
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Create movie sequence with new logic
        final_image_files, durations = create_movie_sequence(
            images_short, images_long, audio_duration,
            long_duration, short_start_duration, short_end_duration, short_acceleration,
            first_image_duration, last_image_duration,
            beat_grid=beat_grid if enable_beat_grid else None,
            random_seed=random_seed,
            long_anchor_delay=long_anchor_delay,
            first_image_override=first_override,
            last_image_override=last_override,
        )
        
        # Process media files (images and videos) with progress bar
        logger.info("Processing media files...")
        processed_clips = []
        for i, (media_path, duration) in enumerate(tqdm(zip(final_image_files, durations), desc="Processing media")):
            if is_video_file(media_path):
                # Process video file - extract first portion and strip audio
                video_clip = process_video_clip(media_path, duration)
                processed_clips.append(video_clip)
                logger.debug(f"Processed video {media_path} for {duration:.2f}s")
            else:
                # Process image file
                processed_img = resize_image_to_standard(media_path)
                # Create ImageClip with specified duration - suppress MoviePy output
                import contextlib
                from io import StringIO
                with contextlib.redirect_stdout(StringIO()):
                    image_clip = ImageClip(processed_img).with_duration(duration)
                processed_clips.append(image_clip)
                logger.debug(f"Processed image {media_path} for {duration:.2f}s")
        
        # Create video clip by concatenating all clips - suppress MoviePy output
        logger.info("Creating video clip...")
        import contextlib
        from io import StringIO
        with contextlib.redirect_stdout(StringIO()):
            video_clip = concatenate_videoclips(processed_clips)
            
            # Trim video to match audio duration exactly
            if video_clip.duration > audio_duration:
                video_clip = video_clip.subclipped(0, audio_duration)
            
            # Combine video and audio
            logger.info("Combining video and audio...")
            final_clip = video_clip.with_audio(audio_clip)
        
        # Write the final movie (keep normal output for progress)
        logger.info(f"Writing movie to {output_path}...")
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        audio_clip.close()
        video_clip.close()
        final_clip.close()
        
        # Clean up individual clips
        for clip in processed_clips:
            clip.close()
        
        logger.info(f"Movie created successfully: {output_path}")
    
    except Exception as e:
        logger.exception("Error creating movie")
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a slideshow movie with beat-aware timing")
    parser.add_argument("--images-short-dir", default="./files/images_short", help="Directory of short images")
    parser.add_argument("--images-long-dir", default="./files/images_long", help="Directory of long images")
    parser.add_argument("--audio-path", default="./files/audio.mp3", help="Audio track path")
    parser.add_argument("--output-path", default="./output_movie.mp4", help="Destination video path")
    
    parser.add_argument("--long-duration", type=float, default=3.0, help="Duration for each long image (seconds)")
    parser.add_argument("--short-start-duration", type=float, default=0.75, help="Starting duration for short images")
    parser.add_argument("--short-end-duration", type=float, default=0.20, help="Ending duration for short images")
    parser.add_argument("--short-acceleration", type=float, default=2.0, help="Acceleration factor for shrinking short durations")
    parser.add_argument("--first-image-duration", type=float, default=2.0, help="Duration for optional 'first' image")
    parser.add_argument("--last-image-duration", type=float, default=2.0, help="Duration for optional 'last' image")
    
    parser.add_argument("--measures-per-long", type=int, default=4, help="Measures between long-image anchors")
    parser.add_argument("--beats-per-measure", type=int, default=4, help="Beats per measure for beat math")
    parser.add_argument("--beat-detection-sensitivity", type=float, default=0.6, help="Onset detection sensitivity (0-1)")
    parser.add_argument("--long-anchor-delay", type=float, default=0.05, help="Delay long images relative to beat (seconds)")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for deterministic sequencing")
    parser.add_argument("--disable-beat-grid", action="store_true", help="Disable beat/onset alignment")
    parser.add_argument("--first-image-path", default=None, help="Explicit path to the opening image")
    parser.add_argument("--last-image-path", default=None, help="Explicit path to the closing image")

    return parser


def main():
    """CLI entry point exposing create_movie configuration as arguments"""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    enable_beat_grid = not args.disable_beat_grid
    
    logger.info("Movie configuration:")
    logger.info("- Long image duration: %.2fs", args.long_duration)
    logger.info("- Short image duration: %.2fs -> %.2fs", args.short_start_duration, args.short_end_duration)
    logger.info("- Short image acceleration: %.2fx", args.short_acceleration)
    logger.info("- First/last durations: %.2fs / %.2fs", args.first_image_duration, args.last_image_duration)
    logger.info("- Beat alignment: %s", "enabled" if enable_beat_grid else "disabled")
    
    create_movie(
        images_short_dir=args.images_short_dir,
        images_long_dir=args.images_long_dir,
        audio_path=args.audio_path,
        output_path=args.output_path,
        long_duration=args.long_duration,
        short_start_duration=args.short_start_duration,
        short_end_duration=args.short_end_duration,
        short_acceleration=args.short_acceleration,
        first_image_duration=args.first_image_duration,
        last_image_duration=args.last_image_duration,
        measures_per_long=args.measures_per_long,
        beats_per_measure=args.beats_per_measure,
        beat_detection_sensitivity=args.beat_detection_sensitivity,
        enable_beat_grid=enable_beat_grid,
        random_seed=args.random_seed,
        long_anchor_delay=args.long_anchor_delay,
        first_image_path=args.first_image_path,
        last_image_path=args.last_image_path,
    )


if __name__ == "__main__":
    main()
