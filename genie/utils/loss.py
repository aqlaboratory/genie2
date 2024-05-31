import torch


def mse(x_pred, x, mask, aggregate=None, eps=1e-10):
    """
    Compute mean squared error.

    Args:
        x_pred:
            [B, N, D] Predicted values.
        x:
            [B, N, D] Groundtruth values.
        mask:
            [B, N] Mask.
        aggregation:
            Aggregation method within each sample, including
                -   None: no aggregation (default)
                -   mean: aggregation by computing mean along second dimension
                -   sum: aggregation by computing sum along second dimension.
        eps:
            Epsilon for computational stability. Default to 1e-10.

    Returns:
        A tensor of mean squared errors, with a shape of [B, N] if no 
        aggregation, or a shape of [B] if using 'mean' or 'sum' aggregation.
    """
    errors = (eps + torch.sum((x_pred - x) ** 2, dim=-1)) ** 0.5
    if aggregate is None:
        return errors * mask
    elif aggregate == 'mean':
        return torch.sum(errors * mask, dim=-1) / torch.sum(mask, dim=-1)
    elif aggregate == 'sum':
        return torch.sum(errors * mask, dim=-1)
    else:
        print('Invalid aggregate method: {}'.format(aggregate))
        exit(0)
