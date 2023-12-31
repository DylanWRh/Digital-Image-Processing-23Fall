import cv2 
import numpy as np
import matplotlib.pyplot as plt


def get_hist_single_channel(img_single_channel, normalized=True):
    assert len(img_single_channel.shape) == 2
    scale = 1
    if normalized:
        h, w = img_single_channel.shape
        scale = h * w
    return cv2.calcHist(
        images=[img_single_channel],
        channels=[0], mask=None, 
        histSize=[256], ranges=[0, 256]
    ).flatten() / scale


def get_hist_multiple_channel(img_multiple_channel, channels=None, normalized=True):
    assert len(img_multiple_channel.shape) == 3
    if channels is None:
        channels = range(img_multiple_channel.shape[-1])
    hists = []
    for i in channels:
        hists.append(get_hist_single_channel(img_multiple_channel[..., i], normalized))
    return hists


def show_histogram(hists, img=None, figsize=None, **kwargs):
    nnum = len(hists)
    if img is not None:
        nnum += 1
    if figsize is None:
        figsize = (5 * nnum, 5)
    
    plt.figure(figsize=figsize)

    if img is not None:
        plt.subplot(1, nnum, 1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(img)
        if 'img_name' in kwargs:
            plt.title(kwargs['img_name'])
    
    flag = (img is not None) + 1
    for i, hist in enumerate(hists):
        plt.subplot(1, nnum, i + flag)
        plt.bar(range(len(hists[i])), hists[i].flatten())
        if 'channel_names' in kwargs:
            plt.title(f"Histogram of channel {kwargs['channel_names'][i]}")


def equalize_hists(hists=None, src_img=None):
    assert (src_img is not None) or (hists is not None)
    if hists is None:
        hists = get_hist_multiple_channel(src_img)
    cdf_hists = np.array([np.cumsum(hist) for hist in hists])
    color_map = np.round(255 * cdf_hists).astype(np.uint8)
    if src_img is None:
        return color_map
    else:
        new_img = np.stack([color_map[i, src_img[:, :, i]] for i in range(len(hists))]).transpose([1, 2, 0])
        return color_map, new_img


def specialize_hists(dst_hists, src_hists=None, src_img=None):
    assert (src_img is not None) or (src_hists is not None)
    if src_hists is None:
        src_hists = get_hist_multiple_channel(src_img)
    EPS = 1e-6 * np.arange(len(src_hists[0]))
    cdf_srcs = np.array([np.cumsum(hist) for hist in src_hists + EPS])
    cdf_dsts = np.array([np.cumsum(hist) for hist in dst_hists + EPS])
    color_map = np.array([
        np.interp(cdf_src, cdf_dst, np.arange(256))
        for cdf_src, cdf_dst in zip(cdf_srcs, cdf_dsts)
    ]).round().astype(np.uint8)
    if src_img is None:
        return color_map
    else:
        new_img = np.stack([color_map[i, src_img[:, :, i]] for i in range(len(src_hists))]).transpose([1, 2, 0])
        return color_map, new_img


def exact_specialize_hists_single_channel(dst_hist, src_img):
    h, w = src_img.shape 
    nnum = h * w 
    temp = []
    src_hist = get_hist_single_channel(src_img)
    modified_img = src_img.copy()
    for i in range(256):
        if src_hist[i] > dst_hist[i]:
            r = np.where(src_img == i)
            abandon_num = len(r[0]) - int(round(nnum * dst_hist[i]))
            abandoned = np.random.choice(len(r[0]), abandon_num, replace=False)
            for ri, rj in zip(r[0][abandoned], r[1][abandoned]):
                temp.append((ri, rj))
    for i in range(256):
        if src_hist[i] < dst_hist[i]:
            r = np.where(src_img == i)
            need_num = int(round(nnum * dst_hist[i])) - len(r[0])
            for (ri, rj) in temp[:need_num]:
                modified_img[ri, rj] = i 
            temp = temp[need_num:]
    return modified_img


def exact_specialize_hists_multiple_channel(dst_hists, src_img):
    modified_img = [
        exact_specialize_hists_single_channel(dst_hist, src_img_channel)
        for dst_hist, src_img_channel in zip(dst_hists, src_img.transpose([2, 0, 1]))
    ]
    return np.stack(modified_img).transpose([1, 2, 0])