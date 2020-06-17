"""
Structure-from-Motion (SfM) Models and wrappers
===============================================

- SfmModel is a torch.nn.Module wrapping both a Depth and a Pose network to enable training in a Structure-from-Motion setup (i.e. from videos)
- SelfSupModel is an SfmModel specialized for self-supervised learning (using videos only)
- SemiSupModel is an SfmModel specialized for semi-supervised learning (using videos and depth supervision)
- ModelWrapper is a torch.nn.Module that wraps an SfmModel to enable easy training and eval with a trainer
- ModelCheckpoint enables saving/restoring state of torch.nn.Module objects

"""
