import numpy as np

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud.
        Input:
            batch_data: BxNx3 array
        Output:
            BxNx3 array with points randomly scaled
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud.
        Input:
            batch_data: BxNx3 array
        Output:
            BxNx3 array with points randomly shifted
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data