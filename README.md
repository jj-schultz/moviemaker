# MontageMaker Owner's Manual

## What is MontageMaker?
MontageMaker turns a folder of photos and one soundtrack into a polished video without any video-editing experience. It automatically resizes every image to full HD, keeps special intro/outro shots in place, and syncs your "long" hero images to the beat so dramatic moments land where the music hits. Short filler images are evenly spaced between those anchors, so the whole slideshow feels intentional rather than random. Point the tool at your folders and you get a 24 fps MP4 with clean audio and professional pacing.

## Sounds awesome!! - How do I use MontageMaker?

### Install MontageMaker on your computer
Follow these steps exactly once per computer. Replace paths as needed.

1. **Download the project**
   ```bash
   git clone https://github.com/jj-schultz/moviemaker.git
   cd moviemaker
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the requirements**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
### Make a montage
1. **Organize your assets**
   - Place fast-cut images/videos in `files/images_short/`
   - Place spotlight images/videos in `files/images_long/`
   - Put the soundtrack at `files/audio.mp3`
   - Optional: add intro/outro frames by naming files `first.*` and `last.*` inside `files/images_long/`, or point to any other files on disk with `--first-image-path` / `--last-image-path`
2. **Generate the montage** using the helper script (all options are optional):
   ```bash
   ./run_movie.sh \
     --images-short-dir ./files/images_short \
     --images-long-dir ./files/images_long \
     --audio-path ./files/audio.mp3 \
     --output-path ./output_movie.mp4 \
     --long-duration 2.5 \
     --measures-per-long 4 \
     --long-anchor-delay 0.05 \
     --random-seed 42
   ```
   The command prints progress bars while it resizes images, aligns beats (unless `--disable-beat-grid` is set), and finally writes the MP4 to the location you chose.

## Full Documentation
Everything MontageMaker does is controlled by command-line switches. Use this section as a quick reference when you need to fine-tune pacing or troubleshoot a run.

### Input & Output
- **`--images-short-dir`** (default `./files/images_short`): Folder for quick-hit filler photos and videos. Keep at least a few files here so gaps can be filled. Videos are automatically trimmed to fit their assigned duration and have audio stripped.
- **`--images-long-dir`** (default `./files/images_long`): Folder for spotlight photos and videos that should stay on screen longer and align with the beat. Videos are processed the same way as short videos.
- **`--audio-path`** (default `./files/audio.mp3`): Background soundtrack in MP3 format. Length of this file sets the movie length.
- **`--output-path`** (default `./output_movie.mp4`): Final MP4 destination. Use an absolute path if you want the file saved elsewhere.

### Timing Controls
- **`--long-duration`** (default `3.0`): Seconds each long image remains visible after the beat.
- **`--short-start-duration`** (default `0.75`): Duration of the very first short image; later shorts shrink from here.
- **`--short-end-duration`** (default `0.20`): Smallest duration allowed for short images near the end of the video.
- **`--short-acceleration`** (default `2.0`): Controls how quickly short durations move from the start value to the end value. Higher numbers shrink faster.
- **`--first-image-duration`** (default `2.0`): How long to hold the optional `first.*` image before regular sequencing begins.
- **`--last-image-duration`** (default `2.0`): How long to display the optional `last.*` image after everything else finishes.
- **`--first-image-path`** (default unset): Absolute or relative path to force as the opening frame (use this if the image lives outside `files/images_long`).
- **`--last-image-path`** (default unset): Absolute or relative path to force as the closing frame.
- **`--long-anchor-delay`** (default `0.05`): Nudges each long image slightly after the detected beat so the music downbeat hits first.

### Beat Grid & Randomness
- **`--measures-per-long`** (default `4`): Number of musical measures between long-image anchors. Lower numbers add more beat-synced hero shots.
- **`--beats-per-measure`** (default `4`): Time-signature helper. Leave at `4` for most pop songs; set to `3` for a waltz feel.
- **`--beat-detection-sensitivity`** (default `0.6`): Higher sensitivity picks up softer beats; lower values ignore weak hits.
- **`--disable-beat-grid`** (flag): Bypass beat detection entirely if the audio is speech-only or you want evenly spaced long images.
- **`--random-seed`** (default unset): Provide a number (e.g., `42`) to make shuffling deterministic so reruns use the same ordering.

### Example Recipes
**Cinematic Slow Burn**
```bash
./run_movie.sh \
  --long-duration 4.0 \
  --short-start-duration 1.5 \
  --short-end-duration 0.75 \
  --measures-per-long 8 \
  --long-anchor-delay 0.1
```

**Hype Reel**
```bash
./run_movie.sh \
  --long-duration 1.5 \
  --short-start-duration 0.4 \
  --short-end-duration 0.15 \
  --short-acceleration 5 \
  --measures-per-long 2 \
  --random-seed 99
```
Tweak any option and rerun the command—the script overwrites the previous MP4 when the same `--output-path` is used.

## I want to write some code and make it better - how do I run the automated tests?
Even if you never touch the code, you can verify everything is working before building a keepsake video.

1. **Activate the same virtual environment** you created earlier: `source .venv/bin/activate`.
2. **Run the lightweight smoke test** (minimal dependencies):
   ```bash
   python -m unittest test_create_movie_simple.py -v
   ```
3. **Run the full test suite** (checks integration points and requires all dependencies):
   ```bash
   pytest -v
   ```
4. **Optional**: target a single module with unittest.
   ```bash
   python -m unittest test_create_movie.py -v
   ```
If a test fails, re-check your Python packages or asset folders, fix the issue, and re-run the same commands until they pass.

## Supported File Formats
### Images
- PNG (.png)
- JPEG (.jpg, .jpeg)

### Videos
- MOV (.mov) - Video files are automatically trimmed to fit their assigned duration and have audio stripped
- MP4 (.mp4) - Same processing as MOV files
- AVI (.avi) - Same processing as MOV files  
- MKV (.mkv) - Same processing as MOV files

**Note**: When using video files, MontageMaker extracts a segment from the middle of each video clip (up to the assigned duration for that slot in the sequence) and removes the audio track. This middle-extraction approach typically provides more representative content than using the beginning, which often contains titles or introductory material. The video content is then resized to fit the target resolution (1920x1080) while maintaining aspect ratio - just like image files, videos are letterboxed with black bars if needed to preserve their original proportions.

## License & Warranty
- MontageMaker ships under the [MIT License](./LICENSE), which allows personal and commercial use, modification, and redistribution.
- The software is provided **as-is** with **no warranty**. Run it at your own risk, make sure you have appropriate ownership of all content, and double-check the generated videos before publishing or sharing them.
