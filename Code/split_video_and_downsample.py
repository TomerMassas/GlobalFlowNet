import os
import sys
import subprocess


def get_video_duration(video_path: str) -> float | None:
    """
    Gets the duration of a video in seconds using ffprobe.

    Args:
        video_path: The path to the video file.

    Returns:
        The duration of the video as a float, or None if an error occurs.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout)
    except FileNotFoundError:
        print("🚨 Error: ffprobe not found. Please ensure ffmpeg is installed and in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"🚨 Error getting video duration: {e.stderr}")
        return None
    except ValueError:
        print("🚨 Error: Could not parse video duration from ffprobe output.")
        return None


def split_video(video_path: str, output_folder: str, INTERVAL_SECONDS:int ,fps: int = 25):
    """
    Splits a video into 20-second segments.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the video segments.
        fps (int): Frame rate (sample rate) for the output segments.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Get video duration to calculate the number of segments
    duration = get_video_duration(video_path)
    if duration is None:
        return

    print(f"🎥 Video: '{video_path}'")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"📂 Output Folder: '{output_folder}'")
    print("-" * 30)

    # Get the base name and extension of the input file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file_extension = os.path.splitext(video_path)[1]

    # Loop through the video and create segments
    for i in range(0, int(duration), INTERVAL_SECONDS):
        start_time = i
        segment_num = (i // INTERVAL_SECONDS) + 1
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
            '-t', str(INTERVAL_SECONDS),
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


if __name__ == "__main__":
    video_path = r"C:\Users\TomerMassas\Videos\downloaded pictime\RAW\199677.mp4"
    output_folder = r"C:\Users\TomerMassas\Desktop\Video project\add clips in album\199677\down_sample"
    FPS = 10
    INTERVAL_SECONDS = 10

    split_video(video_path, output_folder, INTERVAL_SECONDS, FPS)