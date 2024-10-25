import torch

def preprocess_point_features(feature_gaussians, gates):
    """
    Preprocesses the point features by normalizing and sampling them.

    Args:
        feature_gaussians (Tensor): The input feature gaussians.
        gates (Tensor): The gates for conditioning the point features.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the normalized point features and the normalized sampled point features.
    """
    point_features = feature_gaussians.get_point_features
    scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim=-1, p=2) * gates.unsqueeze(0)
    normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim=-1, p=2)
    sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]
    normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim=-1, keepdim=True)

    return normed_point_features, normed_sampled_point_features