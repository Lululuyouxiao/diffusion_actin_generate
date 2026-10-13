import torch


def rearrange(tensor, pattern, **axes):
    if pattern == "b (h c) x y -> b h c (x y)":
        batch, channels, height, width = tensor.shape
        heads = int(axes["h"])
        if channels % heads != 0:
            raise ValueError("Channel count must be divisible by the head count")
        return tensor.reshape(batch, heads, channels // heads, height * width)

    if pattern == "b h (x y) d -> b (h d) x y":
        batch, heads, spatial, dim = tensor.shape
        height = int(axes["x"])
        width = int(axes["y"])
        if spatial != height * width:
            raise ValueError("Spatial size does not match x*y")
        return tensor.permute(0, 1, 3, 2).reshape(batch, heads * dim, height, width)

    if pattern == "b h c (x y) -> b (h c) x y":
        batch, heads, channels, spatial = tensor.shape
        height = int(axes["x"])
        width = int(axes["y"])
        if spatial != height * width:
            raise ValueError("Spatial size does not match x*y")
        return tensor.reshape(batch, heads * channels, height, width)

    if pattern == "b c -> b c 1 1":
        return tensor[:, :, None, None]

    raise NotImplementedError("Unsupported rearrange pattern: {}".format(pattern))
