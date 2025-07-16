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
    return score, [rotation, translation, scale_change]

def downscale_frames(frames):
    """Downscale frames such that the lower dim must be smaller or equal to 640"""
    max_dim = 640
    height, width = frames.shape[-2], frames.shape[-1]
    ratio = width / height

    if height > width:
        new_width = max_dim
        new_height = int(new_width / ratio)
    else:
        new_height = max_dim
        new_width = int(new_height * ratio)

    return torch.nn.functional.interpolate(frames, size=(new_height, new_width), mode='bilinear')

def detect_camera_movement(video_path):
    # Load model
    model = load_model()

    # Load video
    video = VideoReader(video_path, loadAllFrames=True)
    frames = torch.from_numpy(np.transpose(video.getFrames(), (0, 3, 1, 2))) / 255.0
    frames = downscale_frames(frames)

    # Initialize utilities
    shape = [frames.shape[-2], frames.shape[-1]]
    affine_util = AffineUility(shape)

    # Calculate movement scores
    movement_scores = []
    rot, trans,scl = [], [], []

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
            score, coeffs_scores = calculate_movement_score(coeffs[0].cpu().numpy())
            movement_scores.append(score)
            rot.append(coeffs_scores[0])
            trans.append(coeffs_scores[1])
            scl.append(coeffs_scores[2])
            print(f"Frame {i}: Movement Score = {score:.4f}")

            # Add score of 0 for the first frame (no previous frame to compare)
    movement_scores.insert(0, 0.0)
    rot.insert(0, 0.0)
    trans.insert(0, 0.0)
    scl.insert(0, 0.0)

    return movement_scores, rot, trans,scl

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

def subplots_graphs(score, rotation, translation, scale):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(score, label='Movement Score', color='blue')
    axs[0].set_title('Movement Score')
    axs[0].set_ylabel('Score')
    axs[0].grid()

    axs[1].plot(rotation, label='Rotation', color='orange')
    axs[1].set_title('Rotation')
    axs[1].set_ylabel('Rotation (radians)')
    axs[1].grid()

    axs[2].plot(translation, label='Translation', color='green')
    axs[2].set_title('Translation')
    axs[2].set_ylabel('Translation (pixels)')
    axs[2].grid()

    axs[3].plot(scale, label='Scale', color='red')
    axs[3].set_title('Scale Change')
    axs[3].set_xlabel('Time (frames)')
    axs[3].set_ylabel('Scale (log)')
    axs[3].grid()

    plt.tight_layout()
    plt.show()

def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":

    video_dir = r'C:\Users\TomerMassas\Desktop\Video project\add clips in album\44718403'
    # vids_paths = glob(fr'{video_dir}\down_sample\*.mp4')
    # scores = []
    # for video_path in vids_paths:
    #     fps = get_video_fps(video_path)
    #
    #     score_seg, rot_seg, trans_seg, scale_seg = detect_camera_movement(video_path)
    #     subplots_graphs(score_seg, rot_seg, trans_seg, scale_seg)
    #
    #     scores.extend(score_seg)
    #     np.save(os.path.join(video_dir, f'camera_movement_scores_{os.path.basename(video_dir)}_downscale.npy'), np.array(scores))
    #     # plot_scores(scores, fps)



    old_score = np.load(fr"{video_dir}\camera_movement_scores_{os.path.basename(video_dir)}.npy")
    new_score = np.load(fr"{video_dir}\camera_movement_scores_{os.path.basename(video_dir)}_downscale.npy")

    hop_len = 100
    for i in range(0,len(old_score), hop_len):
        plt.figure(figsize=(12, 6))
        plt.plot(old_score[i:i+hop_len], label='Old Movement Score', color='blue')
        plt.axhline(y=40, color='blue', linestyle='--')
        plt.plot(new_score[i:i+hop_len], label='New Movement Score', color='orange')
        plt.axhline(y=20, color='orange', linestyle='--')
        plt.title('Comparison of Old and New Movement Scores')
        plt.xlabel('Frame Index')
        plt.ylabel('Movement Score')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()








