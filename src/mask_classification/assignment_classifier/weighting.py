import pytorch

def jaccard_index(tensor1, tensor2, dim):
    # Only works for int tensors
    intersection = (tensor1 & tensor2).sum(dim=dim)
    union = (tensor1 | tensor2).sum(dim=dim)
    return intersection / union