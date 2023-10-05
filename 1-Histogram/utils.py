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
    ) / scale


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


def equalize_hists(hists, img=None):
    cdf_hists = np.array([np.cumsum(hist) for hist in hists])
    color_map = np.round(255 * cdf_hists).astype(np.uint8)
    if img is None:
        return color_map
    else:
        new_img = np.stack([color_map[i, img[:, :, i]] for i in range(len(hists))]).transpose([1, 2, 0])
        return color_map, new_img


def specialize_hists(src_hists, dst_hists, img=None):
    cdf_srcs = np.array([np.cumsum(hist) for hist in src_hists])
    cdf_dsts = np.array([np.cumsum(hist) for hist in dst_hists])
    color_map = np.array([
        np.interp(cdf_src, cdf_dst, np.arange(256))
        for cdf_src, cdf_dst in zip(cdf_srcs, cdf_dsts)
    ]).round().astype(np.uint8)
    if img is None:
        return color_map
    else:
        new_img = np.stack([color_map[i, img[:, :, i]] for i in range(len(src_hists))]).transpose([1, 2, 0])
        return color_map, new_img