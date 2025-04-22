import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import interp1d


def read_nifti_file(filepath):
    """Read and load data"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def window_transform(volume, windowWidth, windowCenter):
    """
    Return: truncated image according to window center and window width, normalized to [0, 1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    volume = (volume - minWindow) / float(windowWidth)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    volume = volume.astype("float32")

    return volume


def normalize(volume, windowWidth, windowCenter):
    """Data normalization"""
    min = float(windowCenter) - 0.5 * float(windowWidth)
    max = float(windowCenter) + 0.5 * float(windowWidth)  # window_transform
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")

    return volume


def normalize01(volume):
    """Normalize data to range [0, 1]"""
    min_val = np.min(volume)
    max_val = np.max(volume)

    # Normalize the image
    normalized_image = (volume - min_val) / (max_val - min_val)
    return normalized_image


def resize_volume(img, width, height):
    """Resize along the z-axis"""
    # Set the desired depth
    desired_depth = 36
    desired_width = width
    desired_height = height
    # Get the current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Calculate scale factors
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate the image
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize along the z-axis
    # Resampling refers to normalizing voxel sizes in medical images with different sizes
    # Resampling may modify the actual lesion area
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    img = np.flip(img, axis=2)  # Adjust the order

    return img


def resize_image(img, width, height):
    """Resize along the z-axis"""
    # Set the desired depth
    desired_width = width
    desired_height = height
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Adjust dimensions
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1.0
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    img = np.flip(img, axis=2)

    return img


def process_scan(path, width, height):
    """Read and resize the scan data"""
    # Read the scan file
    volume = read_nifti_file(path)
    # Normalize
    if path.find('Jinhua') != -1:
        WW, WL = 400, 60  # Jinhua
    elif path.find('Lihuili_colorectal_cancer_data') != -1:
        WW, WL = 300, 50  # Lihuili
    else:
        WW, WL = 199, 39  # ZJU First Affiliated Hospital
    volume = normalize(volume, WW, WL)
    # Resize width, height, and depth
    volume = resize_volume(volume, width, height)
    return volume


def process_img(path, width, height):
    """Read and resize the scan data"""
    # Read the scan file
    volume = read_nifti_file(path)
    # Normalize
    # if path.find('Jinhua') != -1:
    #     WW, WL = 400, 60  # Jinhua
    # elif path.find('Lihuili_colorectal_cancer_data') != -1:
    #     WW, WL = 300, 50  # Lihuili
    # else:
    #     WW, WL = 199, 39  # ZJU First Affiliated Hospital
    WW, WL = 400, 60  # Unified window width and level
    volume = normalize(volume, WW, WL)
    # Resize width and height
    volume = resize_image(volume, width, height)
    return volume


def process_img_prognosis(volume, path, width, height):
    """Read and resize the scan data"""

    mode_params = {
        "C": (788, 394),
        "T2": (2833, 1417),
        "DWI": (255, 128)
    }

    cur_mode = path.split("/")[-2]

    params = mode_params[cur_mode]
    WW, WL = params

    # Normalize and resize the image
    volume = normalize(volume, WW, WL)
    volume = resize_image(volume, width, height)

    return volume


def process_img01(path, width, height):
    """Read and resize the scan data"""
    # Read the scan file
    volume = read_nifti_file(path)
    volume = normalize01(volume)
    # Resize width and height
    volume = resize_image(volume, width, height)
    return volume


def convert_to_pinyin(name):
    """Convert Chinese to Pinyin"""
    pinyin_list = pinyin(name, style=Style.NORMAL)
    pinyin_name = ' '.join([str(p[0]) for p in pinyin_list])
    return pinyin_name


def interpolate_missing_rects(rects):
    # Extract indices of frames with coordinates and the corresponding coordinates
    indices = [i for i, rect in enumerate(rects) if rect]
    coordinates = np.array([rects[i] for i in indices])

    # Interpolate x1, y1, x2, y2 separately
    interpolated_rects = []
    for dim in range(4):
        # Use 'cubic' spline interpolation, i.e., cubic spline interpolation
        try:
            interp_func = interp1d(indices, coordinates[:, dim], kind='cubic', fill_value='extrapolate')
        except ValueError:
            interp_func = interp1d(indices, coordinates[:, dim], kind='linear', fill_value='extrapolate')
        interpolated_dim = interp_func(range(len(rects)))
        interpolated_rects.append(interpolated_dim)

    # Reconstruct the sequence of interpolated rectangles
    interpolated_rects = np.array(interpolated_rects).T.astype('int')
    return interpolated_rects.tolist()
