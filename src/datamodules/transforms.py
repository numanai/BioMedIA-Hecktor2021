"""Augmentation transforms operating on SimpleITK images.
Credits:
@article{
  kim_deep-cr_2020,
	title = {Deep-{CR} {MTLR}: a {Multi}-{Modal} {Approach} for {Cancer} {Survival} {Prediction} with {Competing} {Risks}},
	shorttitle = {Deep-{CR} {MTLR}},
	url = {https://arxiv.org/abs/2012.05765v1},
	language = {en},
	urldate = {2021-03-16},
	author = {Kim, Sejin and Kazmierski, Michal and Haibe-Kains, Benjamin},
	month = dec,
	year = {2020}
}

"""


import numpy as np
import SimpleITK as sitk
import torch


class ToTensor:
    """Convert a SimpleITK image to torch.Tensor."""
    def __call__(self, image: sitk.Image) -> torch.Tensor:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to convert to tensor.

        Returns
        -------
        torch.Tensor
            The converted tensor.
        """
        array = sitk.GetArrayFromImage(image)
        tensor = torch.from_numpy(array).unsqueeze(0).float()
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomInPlaneRotation:
    """Rotate the image in xy plane by a randomly chosen angle."""
    def __init__(self, max_angle: float, fill_value: float = -1024.):
        """Initialize the transform.

        Parameters
        ----------
        max_angle
            The maximum absolute value of rotation angle.
            The angle of rotation is chosen uniformly at random from
            [-max_angle, max_angle].
        fill_value
            The value used to fill voxels outside image support.
        """
        self.max_angle = max_angle
        self.fill_value = fill_value

    def __call__(self, x: sitk.Image) -> sitk.Image:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        angle = -self.max_angle + 2 * self.max_angle * torch.rand(1).item()
        rotation_centre = np.array(x.GetSize()) / 2
        rotation_centre = x.TransformContinuousIndexToPhysicalPoint(rotation_centre)

        rotation = sitk.Euler3DTransform(
            rotation_centre,
            0,      # the angle of rotation around the x-axis, in radians -> coronal rotation
            0,      # the angle of rotation around the y-axis, in radians -> saggittal rotation
            angle,  # the angle of rotation around the z-axis, in radians -> axial rotation
            (0., 0., 0.)  # no translation
        )
        return sitk.Resample(x, x, rotation, sitk.sitkLinear, self.fill_value)

    def __repr__(self):
        return f"{self.__class__.__name__}(max_angle={self.max_angle}, fill_value={self.fill_value})"


class RandomFlip:
    """Randomly flip an image along a given axis."""
    def __init__(self, dim: int):
        """Initialize the transform.

        Parameters
        ----------
        dim
            The axis along which to flip.
        """
        self.dim = dim
        self.flip_mask = [i == self.dim for i in range(3)]

    def __call__(self, x: sitk.Image) -> sitk.Image:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        if np.random.random() > .5:
            x = sitk.Flip(x, self.flip_mask)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class RandomNoise:
    """Add zero-mean Gaussian noise to an image."""
    def __init__(self, std: float = 1.):
        """Initialize the transform.

        Parameters
        ----------
        std
            The standard deviation of noise.
        """
        self.std = std

    def __call__(self, x: sitk.Image) -> sitk.Image:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        # use Pytorch random generator to be consistent with seeds
        noise = (torch.randn(x.GetSize()[::-1]) * self.std).numpy()
        noise = sitk.GetImageFromArray(noise)
        noise.CopyInformation(x)
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class Normalize:
    """Normalize an image by subtracting the dataset mean and dividing by the
    dataset standard deviation.
    """
    def __init__(self, mean: float, std: float):
        """Initialize the transform.

        Parameters
        ----------
        mean
            The dataset mean.
        std
            The dataset standard deviation.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x: sitk.Image) -> sitk.Image:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        x = (x - self.mean) / self.std

        # division sometimes silently casts the result to sitk.Float64...
        return sitk.Cast(x, sitk.sitkFloat32)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"