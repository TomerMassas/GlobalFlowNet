import os
import numpy as np
import subprocess
import sys



def split_video_by_segments(video_path, time_segments, output_folder, fps=25):

    os.makedirs(output_folder, exist_ok=True)


    # Get the base name and extension of the input file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file_extension = os.path.splitext(video_path)[1]

    # Loop through the video and create segments
    for segment_num, times in enumerate(time_segments):
        start_time, end_time = times
        duration = end_time - start_time
        output_filename = f"{base_name}_segment_{segment_num:03d}{file_extension}"
        output_path = os.path.join(output_folder, output_filename)

        print(f"Processing segment {segment_num}...")

        # Construct the ffmpeg command
        # -ss seeks to the start time (placing it before -i is faster)
        # -t specifies the duration of the segment
        # -r sets the frame rate (re-encodes the video)
        # -an removes the audio track
        # -y overwrites the output file if it exists
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-r', str(fps),
            # '-an',  # No audio
            output_path
        ]

        try:
            # Execute the command, hiding FFMPEG's verbose output on success
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            print("🚨 Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            # If ffmpeg fails, print its error output for debugging
            print(f"🚨 Error creating segment {segment_num}:")
            print(e.stderr.decode())
            continue

    print("\n✅ Video splitting complete!")




