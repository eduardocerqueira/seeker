#date: 2023-11-13T17:03:06Z
#url: https://api.github.com/gists/c7082d83963ae7bd745f6393e43a2d94
#owner: https://api.github.com/users/jfischoff

def frame_diff_with_anchor(video_tensor, anchor_frame_index):
    """
    Compute the frame difference for a video tensor using an anchor frame.
    video_tensor should have shape (batch_size, channels, frames, height, width)
    anchor_frame_index is the index of the anchor frame around which diffs are computed
    """
    # Ensure that the anchor frame is within the correct range
    num_frames = video_tensor.shape[2]
    if not (0 <= anchor_frame_index < num_frames):
        raise ValueError("anchor_frame_index is out of range.")

    # Compute diffs before the anchor frame (if any)
    if anchor_frame_index > 0:
        # split at the anchor
        before_anchor_diff = video_tensor[:, :, :anchor_frame_index] - video_tensor[:, :, 1:anchor_frame_index+1]
    else:
        before_anchor_diff = torch.tensor([]).to(video_tensor.device)

    # Compute diffs after the anchor frame (if any)
    if anchor_frame_index < num_frames - 1:
        # split at the anchor
        after_anchor_diff = video_tensor[:, :, anchor_frame_index+1:] - video_tensor[:, :, anchor_frame_index:-1]
    else:
        after_anchor_diff = torch.tensor([]).to(video_tensor.device)

    anchor = video_tensor[:, :, anchor_frame_index:anchor_frame_index+1]

    diffs = torch.concat([before_anchor_diff, anchor, after_anchor_diff], dim=2)

    return diffs

def video_diff_loss_8(original_video, generated_video, anchor_frame_index):
    """
    Compute the squared frame difference loss for two video tensors using an anchor frame.
    """
    # Compute the frame difference from the anchor frame for both videos
    original_diff = frame_diff_with_anchor(original_video, anchor_frame_index)
    generated_diff = frame_diff_with_anchor(generated_video, anchor_frame_index)

    # Compute the squared difference
    squared_diff = (original_diff - generated_diff)**2

    # Return the mean squared loss over all elements
    return squared_diff.mean()