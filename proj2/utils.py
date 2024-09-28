import numpy as np
import scipy
import skimage as sk

DEFAULT_GAUSSIAN_STD = 3.0

def binarize(m, cutoff):
    ''' binarizes image with cutoff as a percent '''
    clamped_image = (m - np.min(m)) / (np.max(m) - np.min(m))
    return clamped_image > cutoff

def convolve(m, w, mode='same', boundary='fill'):
    ''' convolves m with w, mode is for controlling output dimensions '''
    assert len(m.shape) <= 3
    assert len(w.shape) == 2
    if len(m.shape) == 2:
        return scipy.signal.convolve2d(m, w, mode=mode, boundary=boundary)
    _, _, n = m.shape
    return np.dstack([scipy.signal.convolve2d(m[:,:,i], w, mode=mode, boundary=boundary) for i in range(n)])

def gaussian_kernel(std=DEFAULT_GAUSSIAN_STD):
    ''' constructs a normalized gaussian kernel with the given standard deviation (σ) '''
    # from lecture: sets filter half-width to about 3 σ
    length = int(std * 6)
    gaussian = scipy.signal.windows.gaussian(length, std).reshape(length, 1)
    matrix = np.outer(gaussian, gaussian)
    return matrix / np.sum(matrix)

def unsharp_mask_filter(alpha=1.0, std=DEFAULT_GAUSSIAN_STD):
    ''' the "unsharp" filter which combines sharpening steps into one filter.'''
    gaussian = gaussian_kernel(std)
    return (1 + alpha) * scipy.signal.unit_impulse(gaussian.shape, idx='mid') - alpha * gaussian

def hybrid(imlf, imhf, sigma_lf, sigma_hf, weight_to_hf):
    ''' blends low frequences from imlf and high frequences from imhf '''
    imlf = convolve(imlf, gaussian_kernel(sigma_lf))
    imhf = imhf - convolve(imhf, gaussian_kernel(sigma_hf))
    return imlf * (1 - weight_to_hf) + imhf * (weight_to_hf)

def gaussian_stack(im, n=4, mode='same', boundary='fill', std=DEFAULT_GAUSSIAN_STD):
    stack = [im]
    while len(stack) < n:
        stack.append(convolve(stack[-1], gaussian_kernel(std), mode=mode, boundary=boundary))
    return stack

def laplacian_stack(im, n=4, std=DEFAULT_GAUSSIAN_STD):
    stack = [im]
    while len(stack) < n:
        last_im = stack[-1]
        stack.append(last_im - convolve(last_im, gaussian_kernel(std)))
    return stack

def normalize(m):
    ''' min-max linear normalization (very simple) '''
    return (m - np.min(m)) / (np.max(m) - np.min(m))

def downscale(im, factor=0.5):
    ''' downscale for faster compute '''
    return sk.transform.resize(im, (int(im.shape[0] * factor), int(im.shape[1] * factor)), anti_aliasing=True)

def fft(im):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(sk.color.rgb2gray(im)))))

def circle_mask(center, radius, shape):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (x - center[0])**2 + (y - center[1])**2 <= radius**2

def oval_mask(f1, f2, b, shape):
    y, x = np.ogrid[:shape[0], :shape[1]]
    d1 = np.sqrt((x - f1[0])**2 + (y - f1[1])**2)
    d2 = np.sqrt((x - f2[0])**2 + (y - f2[1])**2)
    return (d1 + d2) <= (2 * b)