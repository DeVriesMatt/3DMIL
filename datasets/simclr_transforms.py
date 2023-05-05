from torchio import transforms
from torchio.transforms import CropOrPad


class ContrastiveTransformations:
    """
    Data augmentation for contrastive learning.
    To allow efficient training, we need to prepare the data loading
    such that we sample two different, random augmentations for each
    image in the batch. The easiest way to do this is by creating a
    transformation that, when being called, applies a set of data
    augmentations to an image twice.
    """

    def __init__(self, base_transforms, n_views=2, img_size=128):
        self.base_transforms = base_transforms
        self.crop_or_pad = CropOrPad(img_size)
        self.n_views = n_views

    def __call__(self, x):
        contrastive = [self.base_transforms(x) for i in range(self.n_views)]
        out = [self.crop_or_pad(c) for c in contrastive]
        return out


contrast_transforms = transforms.Compose(
    [
        transforms.RandomFlip(axes=(0, 1, 2)),
        transforms.RandomAffine(
            scales=(0.9, 1.2),
            degrees=15,
        ),
        transforms.RandomElasticDeformation(
            num_control_points=7,
            max_displacement=7,
        ),
        transforms.RandomNoise(std=(0, 0.1)),
        transforms.RandomBiasField(coefficients=0.1),
        transforms.RandomBlur(std=(0, 0.5)),
        transforms.RandomGamma(log_gamma=(-0.3, 0.3)),
        transforms.RandomMotion(
            degrees=10,
            translation=10,
            num_transforms=2,
        ),
        transforms.RandomSpike(num_spikes=3),
        transforms.RandomGhosting(num_ghosts=3),
        transforms.RandomAnisotropy(),
        transforms.RandomSwap(
            patch_size=10,
            num_iterations=5,
        ),
    ]
)
