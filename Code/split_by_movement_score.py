import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, median_filter
from save_segments import split_video_by_segments
def find_low_movement_segments(movement_score, time_stamps, movement_threshold, min_time_segment):
    """
    Finds time segments where movement scores are below a threshold for a minimum duration.

    Args:
        movement_score (list[float] | np.ndarray): An array of movement scores.
        time_stamps (list[float] | np.ndarray): An array of timestamps for each score.
        movement_threshold (float): The maximum score to be included in a segment.
        min_time_segment (float): The minimum duration for a segment to be valid.

    Returns:
        np.ndarray: A NumPy array of shape (N, 2), where N is the number of valid
                    segments and each row is a [start_time, end_time] pair.
                    Returns an empty array if no segments are found.
    """

    # Ensure inputs are NumPy arrays for vectorized operations
    scores = np.asarray(movement_score)
    times = np.asarray(time_stamps)

    if scores.size != times.size:
        raise ValueError("Input arrays 'movement_score' and 'time_stamps' must have the same length.")

    if scores.size == 0:
        return np.array([])  # Handle empty input

    # 1. Create a boolean mask where scores are below the threshold.
    is_below = scores < movement_threshold

    # 2. Find the start and end indices of all continuous segments at once.
    #    We pad the boolean array with False to correctly handle segments
    #    that start at index 0 or end at the last index.
    bounded_mask = np.concatenate(([False], is_below, [False]))

    # Use np.diff to find where the state changes from False to True (a start)
    # or from True to False (an end).
    diffs = np.diff(bounded_mask.astype(np.int8))

    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0] - 1

    # If no segments exist, return an empty array.
    if start_indices.size == 0:
        return np.array([])

    # 3. Calculate durations for all segments simultaneously.
    start_times = times[start_indices]
    end_times = times[end_indices]
    durations = end_times - start_times

    # 4. Filter out segments that are shorter than the minimum duration.
    is_long_enough = durations >= min_time_segment

    valid_starts = start_times[is_long_enough]
    valid_ends = end_times[is_long_enough]

    # 5. Stack the valid start and end times into a single (N, 2) array.
    return np.stack((valid_starts, valid_ends), axis=1)


if __name__ == "__main__":


    fps = 10
    scores = np.load(r"C:\Users\TomerMassas\Desktop\Video project\add clips in album\199677\camera_movement_scores_199677.npy")
    # scores = minimum_filter(scores, size=5, mode='reflect')
    scores = median_filter(scores, size=5, mode='reflect')
    time = [i / fps for i in range(len(scores))]

    cuts_times = find_low_movement_segments(scores, time, 40, 3)

    #CUT THE VIDEO
    video_file_path = r"C:\Users\TomerMassas\Videos\downloaded pictime\RAW\199677.mp4"
    split_video_by_segments(video_file_path, cuts_times, fr'C:\Users\TomerMassas\Documents\GitHub\GlobalFlowNet\Code\output_clips\{os.path.basename(video_file_path[:-4])}')

    print()



    #split scores and times into intervals of 10 seconds
    cuts = cuts_times.flatten()
    INTERVAL_SECONDS = 10
    scores = np.array(scores)
    time = np.array(time)
    intervals = np.arange(0, len(scores), INTERVAL_SECONDS * fps)
    segments = []
    for start in intervals:
        end = min(start + INTERVAL_SECONDS * fps, len(scores))
        segment_scores = scores[start:end]
        segment_times = time[start:end]

        plt.figure(figsize=(12, 6))
        # add vertical lines for cuts in the time range segment_times
        for cut in cuts:
            if segment_times[0] <= cut <= segment_times[-1]:
                plt.axvline(x=cut, color='red', linestyle='--', label='Cut Point')
        plt.plot(segment_times, segment_scores, label='Movement Score', color='blue')
        plt.title('Camera Movement Detection')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Movement Score')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    print()