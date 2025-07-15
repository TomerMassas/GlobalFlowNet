import os
import json
import numpy as np
import torch
import cv2
from GlobalFlowNets.GlobalPWCNets import getGlobalPWCModel
from Utils.VideoUtility import VideoReader
from Utils.AffineUtility import AffineUility
from glob import glob
import subprocess
import matplotlib.pyplot as plt


def load_model():
    """Load the pre-trained GlobalFlowNet model"""
    # Load configuration
    config = json.load(open('GlobalFlowNets/trainedModels/config.json'))['GlobalNetModelParameters']

    # Load the pre-trained model
    model = getGlobalPWCModel(config, 'GlobalFlowNets/trainedModels/GFlowNet.pth')
    return model.eval().cuda()


def calculate_movement_score(coeffs):
    """Calculate movement score from affine coefficients"""
    # coeffs contains [theta, tx, ty, log_scale]
    rotation = abs(coeffs[0])  # rotation angle
    translation = np.sqrt(coeffs[1] ** 2 + coeffs[2] ** 2)  # translation magnitude
    scale_change = abs(coeffs[3])  # scale change (log scale)

    # Combined movement score (you can adjust weights as needed)
    score = rotation * 100 + translation + scale_change * 10
    return score


def detect_camera_movement(video_path):
    # Load model
    model = load_model()

    # Load video
    video = VideoReader(video_path, loadAllFrames=True)
    frames = torch.from_numpy(np.transpose(video.getFrames(), (0, 3, 1, 2))) / 255.0

    # Initialize utilities
    shape = [frames.shape[-2], frames.shape[-1]]
    affine_util = AffineUility(shape)

    # Calculate movement scores
    movement_scores = []

    with torch.no_grad():
        for i in range(1, frames.shape[0]):
            # Get consecutive frames
            frame1 = frames[i - 1:i].cuda()
            frame2 = frames[i:i + 1].cuda()

            # Estimate optical flow
            flow = model.estimateFlowFull(frame1, frame2)

            # Convert flow to affine coefficients
            coeffs = affine_util.getFlowCoeffs(flow)

            # Calculate movement score
            score = calculate_movement_score(coeffs[0].cpu().numpy())
            movement_scores.append(score)
            print(f"Frame {i}: Movement Score = {score:.4f}")

            # Add score of 0 for the first frame (no previous frame to compare)
    movement_scores.insert(0, 0.0)


    return movement_scores

def print_fps(video_path):
    """
    Print the frames per second (FPS) of the video.

    Args:
        video_path: Path to input video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps:.2f}")
    cap.release()
    return fps

def get_video_fps(video_path: str) -> float | None:

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',  # Select the first video stream
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    fps_fraction = result.stdout.strip()

    # The FPS is often returned as a fraction (e.g., "30/1" or "2997/100")
    if '/' in fps_fraction:
        num, den = map(int, fps_fraction.split('/'))
        return num / den if den != 0 else 0
    else:
        return float(fps_fraction)

def plot_scores(scores, fps):

    # Calculate time in seconds for each score
    time = [i / fps for i in range(len(scores))]

    plt.figure(figsize=(12, 6))
    plt.plot(time, scores, label='Movement Score', color='blue')
    plt.title('Camera Movement Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Movement Score')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()




if __name__ == "__main__":

    video_dir = r'C:\Users\TomerMassas\Desktop\Video project\add clips in album\199677'
    vids_paths = glob(fr'{video_dir}\down_sample\*.mp4')
    scores = []
    for video_path in vids_paths:
        fps = get_video_fps(video_path)

        score_seg = detect_camera_movement(video_path)
        scores.extend(score_seg)
        np.save(os.path.join(video_dir, f'camera_movement_scores_{os.path.basename(video_dir)}.npy'), np.array(scores))
        plot_scores(scores, fps)


    plot_scores(scores, fps)





