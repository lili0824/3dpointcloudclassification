import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each pair of points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point squared distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape

    # Compute squared distances
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)  # [B, 1, M]

    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    # print(f"points shape: {points.shape}")  # Should be [B, N, C]
    # print(f"idx shape: {idx.shape}")  # Should be [B, S]
    # print(f"batch_indices shape: {batch_indices.shape}")  # Should be [B, S]
    # print(f"new_points shape: {new_points.shape}")

    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    # xyz = xyz.transpose(1, 2)
    # print(f"xyz shape after transpose: {xyz.shape}")
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.float()
    # print(f"xyz shape: {xyz.shape}")
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute squared distances between all points and query points
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    # Initialize group_idx with all indices
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]

    # Mask points outside the radius
    group_idx[sqrdists > radius ** 2] = N  # Mark points outside the radius with N (invalid index)

    # Sort group_idx based on distances and keep the closest nsample points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]

    # Handle cases where fewer than nsample points are within the radius
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)  # [B, S, nsample]
    mask = group_idx >= N  # Mask for invalid indices (>= N to handle edge cases)

    # Ensure mask and group_idx have the same shape
    if mask.shape != group_idx.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match group_idx shape {group_idx.shape}")

    group_idx[mask] = group_first[mask]  # Replace invalid indices with the first valid index

    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)

        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] or [B, N, C]
            points: input points data, [B, D, N] or [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        # Ensure xyz and points have the correct shape [B, N, C] and [B, N, D]
        if xyz.shape[-1] == 3:  # If xyz is [B, N, C], no need to transpose
            pass
        else:  # If xyz is [B, C, N], transpose it to [B, N, C]
            xyz = xyz.permute(0, 2, 1)

        if points is not None:
            if points.shape[-1] == xyz.shape[-1]:  # If points is [B, N, D], no need to transpose
                pass
            else:  # If points is [B, D, N], transpose it to [B, N, D]
                points = points.permute(0, 2, 1)

        xyz = xyz.float()
        if points is not None:
            points = points.float()
        # Extract batch size, number of points, and number of coordinates
        B, N, C = xyz.shape
        S = self.npoint

        # Sample points using farthest_point_sample
        sampled_indices = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, sampled_indices)

        new_points_list = []

        # Iterate over radius and nsample lists
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            group_idx = query_ball_point(radius, K, xyz, new_xyz)

            # Index points
            grouped_xyz = index_points(xyz, group_idx)

            # Normalize grouped_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)

            if points is not None:
                grouped_points = index_points(points, group_idx)

                # Concatenate grouped points and grouped_xyz
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            # Permute for convolution
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            # Apply convolution and batch normalization
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pooling
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

            new_points_list.append(new_points)

        # Permute new_xyz back to [B, C, S]
        new_xyz = new_xyz.permute(0, 2, 1)

        # Concatenate all new points
        new_points_concat = torch.cat(new_points_list, dim=1)

        return new_xyz, new_points_concat

    def forward_debug(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] or [B, N, C]
            points: input points data, [B, D, N] or [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        # Debug: Initial shapes
        # print(f"Initial xyz shape: {xyz.shape}")  # Should be [B, C, N] or [B, N, C]
        # if points is not None:
        #     print(f"Initial points shape: {points.shape}")  # Should be [B, D, N] or [B, N, D]

        # Ensure xyz and points have the correct shape [B, N, C] and [B, N, D]
        if xyz.shape[-1] == 3:  # If xyz is [B, N, C], no need to transpose
            print("xyz is already in the correct shape [B, N, C].")
        else:  # If xyz is [B, C, N], transpose it to [B, N, C]
            xyz = xyz.permute(0, 2, 1)
            print(f"xyz shape after transpose: {xyz.shape}")  # Should be [B, N, C]

        if points is not None:
            if points.shape[-1] == xyz.shape[-1]:  # If points is [B, N, D], no need to transpose
                print("points is already in the correct shape [B, N, D].")
            else:  # If points is [B, D, N], transpose it to [B, N, D]
                points = points.permute(0, 2, 1)
                print(f"points shape after transpose: {points.shape}")  # Should be [B, N, D]

        # Extract batch size, number of points, and number of coordinates
        B, N, C = xyz.shape
        S = self.npoint

        # Debug: Sampling points
        print(f"Sampling {S} points using farthest_point_sample...")
        sampled_indices = farthest_point_sample(xyz, S)
        print(f"sampled_indices shape: {sampled_indices.shape}")  # Should be [B, S]

        new_xyz = index_points(xyz, sampled_indices)
        print(f"new_xyz shape after sampling: {new_xyz.shape}")  # Should be [B, S, C]

        new_points_list = []

        # Iterate over radius and nsample lists
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            print(f"\nIteration {i+1}: Processing radius {radius} and nsample {K}...")

            # Debug: Shapes before query_ball_point
            print(f"Iteration {i+1}: xyz shape before query_ball_point: {xyz.shape}")  # Should be [B, N, C]
            print(f"Iteration {i+1}: new_xyz shape before query_ball_point: {new_xyz.shape}")  # Should be [B, S, C]

            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            print(f"Iteration {i+1}: group_idx shape: {group_idx.shape}")  # Should be [B, S, K]

            # Debug: Shapes after query_ball_point
            print(f"Iteration {i+1}: xyz shape after query_ball_point: {xyz.shape}")  # Should still be [B, N, C]
            print(f"Iteration {i+1}: new_xyz shape after query_ball_point: {new_xyz.shape}")  # Should still be [B, S, C]

            # Index points
            grouped_xyz = index_points(xyz, group_idx)
            print(f"Iteration {i+1}: grouped_xyz shape: {grouped_xyz.shape}")  # Should be [B, S, K, C]

            # Normalize grouped_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            print(f"Iteration {i+1}: grouped_xyz shape after normalization: {grouped_xyz.shape}")  # Should be [B, S, K, C]

            if points is not None:
                grouped_points = index_points(points, group_idx)
                print(f"Iteration {i+1}: grouped_points shape: {grouped_points.shape}")  # Should be [B, S, K, D]

                # Concatenate grouped points and grouped_xyz
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                print(f"Iteration {i+1}: grouped_points shape after concatenation: {grouped_points.shape}")  # Should be [B, S, K, D+C]
            else:
                grouped_points = grouped_xyz

            # Permute for convolution
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            print(f"Iteration {i+1}: grouped_points shape after permute: {grouped_points.shape}")  # Should be [B, D, K, S]

            # Apply convolution and batch normalization
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
                print(f"Iteration {i+1}: grouped_points shape after conv {j+1}/{len(self.conv_blocks[i])}: {grouped_points.shape}")

            # Max pooling
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            print(f"Iteration {i+1}: new_points shape after max pooling: {new_points.shape}")  # Should be [B, D', S]

            new_points_list.append(new_points)

        # Permute new_xyz back to [B, C, S]
        new_xyz = new_xyz.permute(0, 2, 1)
        print(f"new_xyz shape after permute: {new_xyz.shape}")  # Should be [B, C, S]

        # Concatenate all new points
        new_points_concat = torch.cat(new_points_list, dim=1)
        print(f"new_points_concat shape: {new_points_concat.shape}")  # Should be [B, D', S]

        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

