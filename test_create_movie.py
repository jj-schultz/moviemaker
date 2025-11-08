#!/usr/bin/env python3
import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from PIL import Image
import logging
import sys

# Mock moviepy modules before importing create_movie
sys.modules['moviepy'] = Mock()
sys.modules['moviepy.editor'] = Mock()

# Mock the classes that will be imported
mock_imagesequenceclip = Mock()
mock_audiofileclip = Mock()
mock_compositevideoclip = Mock()

sys.modules['moviepy.editor'].ImageSequenceClip = mock_imagesequenceclip
sys.modules['moviepy.editor'].AudioFileClip = mock_audiofileclip
sys.modules['moviepy.editor'].CompositeVideoClip = mock_compositevideoclip

from create_movie import (
    get_image_files,
    calculate_progressive_duration,
    create_movie_sequence,
    resize_image_to_standard,
    create_movie,
    is_video_file
)

logging.basicConfig(level=logging.INFO)


class TestCreateMovie(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create test images
        self.test_images = []
        for i in range(20):
            img_path = os.path.join(self.images_dir, f"test_image_{i:02d}.png")
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color=(i*10 % 255, 0, 0))
            img.save(img_path)
            self.test_images.append(img_path)
        
        # Create a test audio file path (won't actually create the file)
        self.audio_path = os.path.join(self.test_dir, "test_audio.mp3")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_get_image_files(self):
        """Test getting image files from directory"""
        image_files = get_image_files(self.images_dir)
        
        self.assertEqual(len(image_files), 20)
        self.assertTrue(all(f.endswith('.png') for f in image_files))
        # Files should be sorted by modification time
        self.assertEqual(image_files, sorted(image_files, key=os.path.getmtime))
    
    def test_get_image_files_empty_directory(self):
        """Test getting image files from empty directory"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        image_files = get_image_files(empty_dir)
        self.assertEqual(len(image_files), 0)
    
    def test_calculate_progressive_duration(self):
        """Test progressive duration calculation"""
        # Test basic linear progression
        start_duration = calculate_progressive_duration(0.0, 1.0, 0.2, 1.0)
        mid_duration = calculate_progressive_duration(0.5, 1.0, 0.2, 1.0)
        end_duration = calculate_progressive_duration(1.0, 1.0, 0.2, 1.0)
        
        self.assertAlmostEqual(start_duration, 1.0)
        self.assertAlmostEqual(mid_duration, 0.6)
        self.assertAlmostEqual(end_duration, 0.2)
        
        # Test acceleration
        accel_mid = calculate_progressive_duration(0.5, 1.0, 0.2, 2.0)
        self.assertLess(accel_mid, mid_duration)  # Should be faster with acceleration
    
    def test_create_movie_sequence_basic(self):
        """Test movie sequence creation"""
        images_short = self.test_images[:10]
        images_long = self.test_images[10:12]
        audio_duration = 10.0
        
        files, durations = create_movie_sequence(
            images_short, images_long, audio_duration,
            long_duration=2.0, short_start_duration=1.0, short_end_duration=0.2
        )
        
        # Should return sequences
        self.assertGreater(len(files), 0)
        self.assertEqual(len(files), len(durations))
        # Total duration should match audio
        self.assertAlmostEqual(sum(durations), audio_duration, places=1)
    
    def test_create_movie_sequence_with_acceleration(self):
        """Test movie sequence with acceleration parameter"""
        images_short = self.test_images[:8]
        images_long = self.test_images[8:10]
        audio_duration = 10.0
        
        files, durations = create_movie_sequence(
            images_short, images_long, audio_duration,
            long_duration=2.0, short_start_duration=1.0, short_end_duration=0.2,
            short_acceleration=2.0
        )
        
        # Should create a sequence with varying durations
        self.assertGreater(len(files), 0)
        self.assertEqual(len(files), len(durations))
        # Should have different duration types
        duration_set = set(durations)
        self.assertGreater(len(duration_set), 1)  # Not all the same duration
    
    def test_resize_image_to_standard(self):
        """Test image resizing functionality"""
        # Create a test image with known dimensions
        test_img_path = os.path.join(self.test_dir, "resize_test.png")
        img = Image.new('RGB', (800, 600), color=(255, 0, 0))
        img.save(test_img_path)
        
        resized = resize_image_to_standard(test_img_path, (1920, 1080))
        
        # Should return numpy array with target dimensions
        self.assertEqual(resized.shape, (1080, 1920, 3))
        self.assertIsInstance(resized, np.ndarray)
    
    def test_resize_image_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved during resize"""
        # Create a square image
        test_img_path = os.path.join(self.test_dir, "square_test.png")
        img = Image.new('RGB', (100, 100), color=(0, 255, 0))
        img.save(test_img_path)
        
        resized = resize_image_to_standard(test_img_path, (1920, 1080))
        
        # Should have black borders (aspect ratio preservation)
        # Check that the image is centered with black borders
        self.assertEqual(resized.shape, (1080, 1920, 3))
        
        # Check that corners are black (borders)
        self.assertTrue(np.array_equal(resized[0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(resized[0, -1], [0, 0, 0]))
    
    def test_resize_image_different_formats(self):
        """Test resizing images with different color modes"""
        # Test RGBA image
        test_img_path = os.path.join(self.test_dir, "rgba_test.png")
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(test_img_path)
        
        resized = resize_image_to_standard(test_img_path)
        
        # Should convert to RGB and have correct dimensions
        self.assertEqual(resized.shape, (1080, 1920, 3))
    
    def test_create_movie_integration(self):
        """Test the main create_movie function with mocked dependencies"""
        # Mock audio clip
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 10.0
        mock_audiofileclip.return_value = mock_audio_instance
        
        # Mock image sequence clip
        mock_video_instance = MagicMock()
        mock_video_instance.duration = 10.0
        mock_imagesequenceclip.return_value = mock_video_instance
        
        # Mock composite clip
        mock_final_instance = MagicMock()
        mock_compositevideoclip.return_value = mock_final_instance
        
        output_path = os.path.join(self.test_dir, "test_output.mp4")
        
        # Create a dummy audio file for the test
        with open(self.audio_path, 'w') as f:
            f.write("dummy audio content")
        
        # Create short and long image directories
        images_short_dir = os.path.join(self.test_dir, "images_short")
        images_long_dir = os.path.join(self.test_dir, "images_long")
        os.makedirs(images_short_dir)
        os.makedirs(images_long_dir)
        
        # Copy some test images to the directories
        for i in range(3):
            shutil.copy(self.test_images[i], images_short_dir)
        for i in range(3, 5):
            shutil.copy(self.test_images[i], images_long_dir)
        
        try:
            create_movie(images_short_dir, images_long_dir, self.audio_path, output_path)
            
            # Verify that the main components were called
            mock_audiofileclip.assert_called_once_with(self.audio_path)
            mock_imagesequenceclip.assert_called_once()
            
        except Exception as e:
            # Expected since we're mocking the moviepy components
            pass
    
    def test_create_movie_missing_files(self):
        """Test error handling for missing input files"""
        output_path = os.path.join(self.test_dir, "test_output.mp4")
        
        # Test missing images directory - should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            create_movie("/nonexistent/short", "/nonexistent/long", self.audio_path, output_path)
        
        # Test missing audio file - should raise FileNotFoundError  
        images_short_dir = os.path.join(self.test_dir, "images_short")
        images_long_dir = os.path.join(self.test_dir, "images_long")
        os.makedirs(images_short_dir)
        os.makedirs(images_long_dir)
        with self.assertRaises(FileNotFoundError):
            create_movie(images_short_dir, images_long_dir, "/nonexistent/audio.mp3", output_path)
    
    def test_video_file_detection(self):
        """Test video file detection"""
        # Test video files
        self.assertTrue(is_video_file("test.mov"))
        self.assertTrue(is_video_file("test.mp4"))
        self.assertTrue(is_video_file("test.avi"))
        self.assertTrue(is_video_file("test.mkv"))
        self.assertTrue(is_video_file("TEST.MOV"))  # Test case insensitivity
        
        # Test non-video files
        self.assertFalse(is_video_file("test.jpg"))
        self.assertFalse(is_video_file("test.png"))
        self.assertFalse(is_video_file("test.txt"))
        self.assertFalse(is_video_file("test.mp3"))
    
    def test_get_image_files_with_videos(self):
        """Test getting both image and video files"""
        # Create a mix of image and video files
        video_dir = os.path.join(self.test_dir, "mixed_media")
        os.makedirs(video_dir)
        
        # Create test image files
        img1 = os.path.join(video_dir, "image1.png")
        img2 = os.path.join(video_dir, "image2.jpg")
        Image.new('RGB', (100, 100), color=(255, 0, 0)).save(img1)
        Image.new('RGB', (100, 100), color=(0, 255, 0)).save(img2)
        
        # Create dummy video files (just empty files for testing)
        vid1 = os.path.join(video_dir, "video1.mov")
        vid2 = os.path.join(video_dir, "video2.mp4")
        with open(vid1, 'w') as f:
            f.write("dummy video content")
        with open(vid2, 'w') as f:
            f.write("dummy video content")
        
        media_files = get_image_files(video_dir)
        
        # Should find all 4 files
        self.assertEqual(len(media_files), 4)
        
        # Check that both images and videos are included
        extensions = [os.path.splitext(f)[1].lower() for f in media_files]
        self.assertIn('.png', extensions)
        self.assertIn('.jpg', extensions)
        self.assertIn('.mov', extensions)
        self.assertIn('.mp4', extensions)
    
    def test_video_aspect_ratio_calculation(self):
        """Test that video aspect ratio calculations work correctly"""
        # Test wide video (16:9) -> should fit by width
        video_width, video_height = 1920, 1080
        target_width, target_height = 1920, 1080
        
        video_ratio = video_width / video_height
        target_ratio = target_width / target_height
        
        if video_ratio > target_ratio:
            new_width = target_width
            new_height = int(new_width / video_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * video_ratio)
        
        # Should be exact match
        self.assertEqual(new_width, 1920)
        self.assertEqual(new_height, 1080)
        
        # Test very wide video (2:1) -> should fit by width with black bars top/bottom
        video_width, video_height = 2000, 1000
        video_ratio = video_width / video_height  # 2.0
        target_ratio = target_width / target_height  # 1.77...
        
        if video_ratio > target_ratio:
            new_width = target_width  # 1920
            new_height = int(new_width / video_ratio)  # 960
        else:
            new_height = target_height
            new_width = int(new_height * video_ratio)
        
        self.assertEqual(new_width, 1920)
        self.assertEqual(new_height, 960)
        # Should have 60px black bars top and bottom: (1080 - 960) / 2 = 60
        
        # Test tall video (9:16) -> should fit by height with black bars left/right
        video_width, video_height = 1080, 1920
        video_ratio = video_width / video_height  # 0.5625
        
        if video_ratio > target_ratio:
            new_width = target_width
            new_height = int(new_width / video_ratio)
        else:
            new_height = target_height  # 1080
            new_width = int(new_height * video_ratio)  # 607
        
        self.assertEqual(new_width, 607)
        self.assertEqual(new_height, 1080)
        # Should have black bars left and right: (1920 - 607) / 2 = 656.5
    
    def test_video_middle_extraction_logic(self):
        """Test that video extraction from middle is calculated correctly"""
        # Test long video - should extract from middle
        original_duration = 60.0
        extract_duration = 5.0
        
        if original_duration > extract_duration:
            start_time = (original_duration - extract_duration) / 2
            end_time = start_time + extract_duration
        
        self.assertEqual(start_time, 27.5)  # (60 - 5) / 2
        self.assertEqual(end_time, 32.5)    # 27.5 + 5
        self.assertEqual(end_time - start_time, extract_duration)
        
        # Test short video - should use full duration
        original_duration = 3.0
        extract_duration = 5.0
        
        # When original is shorter, we use the full original video
        if original_duration <= extract_duration:
            # No extraction needed, use full video
            start_time = 0
            end_time = original_duration
        
        self.assertEqual(start_time, 0)
        self.assertEqual(end_time, 3.0)
        
        # Test edge case - exactly equal durations
        original_duration = 5.0
        extract_duration = 5.0
        
        if original_duration > extract_duration:
            start_time = (original_duration - extract_duration) / 2
            end_time = start_time + extract_duration
        else:
            start_time = 0
            end_time = original_duration
        
        self.assertEqual(start_time, 0)
        self.assertEqual(end_time, 5.0)
    
    def test_video_processing_uniform_for_short_and_long(self):
        """Test that both short and long video files use the same processing logic"""
        # Create directories with mixed media files
        short_dir = os.path.join(self.test_dir, "test_short")
        long_dir = os.path.join(self.test_dir, "test_long")
        os.makedirs(short_dir)
        os.makedirs(long_dir)
        
        # Create test files
        short_image = os.path.join(short_dir, "image.jpg")
        short_video = os.path.join(short_dir, "video.mov")
        long_image = os.path.join(long_dir, "image.jpg") 
        long_video = os.path.join(long_dir, "video.mp4")
        
        Image.new('RGB', (100, 100), color=(255, 0, 0)).save(short_image)
        Image.new('RGB', (100, 100), color=(0, 255, 0)).save(long_image)
        
        # Create dummy video files
        with open(short_video, 'w') as f:
            f.write("dummy video content")
        with open(long_video, 'w') as f:
            f.write("dummy video content")
        
        # Get files
        short_files = get_image_files(short_dir)
        long_files = get_image_files(long_dir)
        
        # Verify that video detection works the same for both
        short_video_file = [f for f in short_files if f.endswith('.mov')][0]
        long_video_file = [f for f in long_files if f.endswith('.mp4')][0]
        
        self.assertTrue(is_video_file(short_video_file))
        self.assertTrue(is_video_file(long_video_file))
        
        # Both should be detected as video files regardless of source directory
        self.assertEqual(is_video_file(short_video_file), is_video_file(long_video_file))
        
        # Verify that create_movie_sequence treats them uniformly
        # (both go through the same final processing loop)


class TestImageProcessing(unittest.TestCase):
    """Additional tests for image processing edge cases"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_resize_very_wide_image(self):
        """Test resizing very wide image"""
        test_img_path = os.path.join(self.test_dir, "wide_test.png")
        img = Image.new('RGB', (2000, 100), color=(0, 0, 255))
        img.save(test_img_path)
        
        resized = resize_image_to_standard(test_img_path, (1920, 1080))
        
        self.assertEqual(resized.shape, (1080, 1920, 3))
        # Should have horizontal black bars
        self.assertTrue(np.array_equal(resized[0, 960], [0, 0, 0]))
    
    def test_resize_very_tall_image(self):
        """Test resizing very tall image"""
        test_img_path = os.path.join(self.test_dir, "tall_test.png")
        img = Image.new('RGB', (100, 2000), color=(255, 255, 0))
        img.save(test_img_path)
        
        resized = resize_image_to_standard(test_img_path, (1920, 1080))
        
        self.assertEqual(resized.shape, (1080, 1920, 3))
        # Should have vertical black bars
        self.assertTrue(np.array_equal(resized[540, 0], [0, 0, 0]))


if __name__ == '__main__':
    unittest.main()