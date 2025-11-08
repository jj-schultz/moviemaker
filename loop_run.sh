#!/bin/bash

for i in {1..10}; do
  ./run_movie.sh \
    --images-short-dir ./files/images_short \
    --images-long-dir ./files/images_long \
    --audio-path ./files/audio.mp3 \
    --output-path ~/Desktop/montage_${i}.mp4 \
    --first-image-duration 2.5 \
    --long-duration 1.5 \
    --measures-per-long 2 \
    --long-anchor-delay 0.05 \
    --short-start-duration 0.75 \
    --short-acceleration 7
done
