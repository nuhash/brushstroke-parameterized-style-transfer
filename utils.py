from tqdm import tqdm
import requests
import os
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage import segmentation
from scipy.spatial import ConvexHull
import scipy.spatial
import matplotlib.pyplot as plt
import cv2
import skimage

#------------------------------------------------------------------
# I/O
#------------------------------------------------------------------

def download_weights(url, name):
    """
    Downloads the checkpoint file specified by 'url'.

    Args:
        url (str): URL specifying the checkpoint file.
        name (str): Name under which the checkpoint file will be stored.

    Returns:
        (str): Path to the checkpoint file.
    """
    ckpt_dir = 'pretrained_weights'
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


#------------------------------------------------------------------
# Brushstrokes
#------------------------------------------------------------------

def clusters_to_strokes(segments, img, H, W, sec_scale=0.001, width_scale=1,init_prob=None,offset=0.5,init_width=None,gradthresh=0.08):
    dx = cv2.Sobel(np.array(img),cv2.CV_64F,1,0,ksize=11,scale=2**(2+1+0-11*2))
    dy = cv2.Sobel(np.array(img),cv2.CV_64F,0,1,ksize=11,scale=2**(2+0+1-11*2))
    mag = np.sqrt(dx**2+dy**2)
    dxn = dx/(mag+0.0001)
    dyn = dy/(mag+0.0001)
    
    fx = np.max(np.abs(dx),-1)
    fy = np.max(np.abs(dy),-1)
    fm = fx+fy
    blurfm = skimage.filters.gaussian(fm)
    
    sorted_vals = np.sort(blurfm.flatten())
    norm_cdf = scipy.stats.norm.cdf(sorted_vals)
    norm_cdf = norm_cdf-np.min(norm_cdf)
    norm_cdf = norm_cdf/np.max(norm_cdf)
    threshold = sorted_vals[np.argmin(np.abs(norm_cdf-gradthresh))]
    
    points = []
    stride = 8
    xp = np.arange(0,fm.shape[0],stride)
    yp = np.arange(0,fm.shape[1],stride)
    fm = fm/np.max(fm)
    for x in xp:
      for y in yp:
        lmax = np.argmax(fm[x:x+stride,y:y+stride]).astype(np.int)
        xoff = lmax//stride
        yoff = lmax%stride
        if x+xoff>=fm.shape[0] or y+yoff>=fm.shape[1]:
          continue
        if fm[x+xoff,y+yoff]>threshold:
          points.append([x+xoff,y+yoff,np.mean(dxn[x+xoff,y+yoff]),np.mean(dyn[x+xoff,y+yoff])])
    points = np.array(points)
    tensors = np.zeros((points.shape[0],2,2))
    theta = np.arctan2(points[:,2],points[:,3])
    tensors[:,0,0] = np.cos(2*theta)
    tensors[:,0,1] = np.sin(2*theta)
    tensors[:,1,0] = np.sin(2*theta)
    tensors[:,1,1] = -np.cos(2*theta)
    
    segments += -np.abs(np.min(segments))
    num_clusters = np.max(segments)                                                                                                     
    clusters_params = {'center': [],
                       's': [],
                       'e': [],
                       'bp1': [],
                       'bp2': [],
                       'num_pixels': [],
                       'stddev': [],
                       'width': [],
                       'color_rgb': []
                       }
#     if init_prob is not None:
#         sorted_vals = np.sort(init_prob.flatten())
#         norm_cdf = scipy.stats.norm.cdf(sorted_vals)
#         norm_cdf = norm_cdf-np.min(norm_cdf)
#         norm_cdf = norm_cdf/np.max(norm_cdf)
    
    for cluster_idx in range(num_clusters + 1):
        cluster_mask = segments==cluster_idx
        if np.sum(cluster_mask) < 5: continue
        cluster_mask_nonzeros = np.nonzero(cluster_mask)

        cluster_points = np.stack((cluster_mask_nonzeros[0], cluster_mask_nonzeros[1]), axis=-1)
        try:
            convex_hull = ConvexHull(cluster_points)
        except:
            continue

        # find the two points (pixels) in the cluster that have the largest distance between them
        border_points = cluster_points[convex_hull.simplices.reshape(-1)]
        dist = np.sum((np.expand_dims(border_points, axis=1) - border_points)**2, axis=-1)
        max_idx_a, max_idx_b = np.nonzero(dist == np.max(dist))
        point_a = border_points[max_idx_a[0]]
        point_b = border_points[max_idx_b[0]]
        # compute the two intersection points of the line that goes orthogonal to point_a and point_b
        v_ba = point_b - point_a
        v_orth = np.array([v_ba[1], -v_ba[0]])
        m = (point_a + point_b) / 2.0
        n = m + 0.5 * v_orth
        p = cluster_points[convex_hull.simplices][:, 0]
        q = cluster_points[convex_hull.simplices][:, 1]
        u = - ((m[..., 0] - n[..., 0]) * (m[..., 1] - p[..., 1]) - (m[..., 1] - n[..., 1]) * (m[..., 0] - p[..., 0])) \
            / ((m[..., 0] - n[..., 0]) * (p[..., 1] - q[..., 1]) - (m[..., 1] - n[..., 1]) * (p[..., 0] - q[..., 0]))
        intersec_idcs = np.logical_and(u >= 0, u <= 1)
        intersec_points = p + u.reshape(-1, 1) * (q - p)
        intersec_points = intersec_points[intersec_idcs]
        
        if init_width == None:
            width = np.sum((intersec_points[0] - intersec_points[1])**2)
        else:
            width = init_width
        
        if width == 0.0: continue

        center_x = np.mean(cluster_mask_nonzeros[0]) 
        center_y = np.mean(cluster_mask_nonzeros[1]) 
        
        dists = np.sqrt(np.sum((points[:,:2]-np.array([[center_x,center_y]]))**2,-1))
        weights = np.exp(-dists)/np.sum(np.exp(-dists))
        localtensor = np.sum(tensors*np.reshape(weights,(points.shape[0],1,1)),0)
        w,v = np.linalg.eig(localtensor)
        major = np.argmax(w)
        
#         if init_prob is not None:
#             content_error = np.max(init_prob[cluster_mask_nonzeros[0].astype(np.int),cluster_mask_nonzeros[1].astype(np.int)])
#             prob_keep = norm_cdf[(np.abs(sorted_vals - content_error)).argmin()]
#             if offset>prob_keep:
#                 continue
        center_point = np.array([center_x, center_y])
        ori = v[::-1,major]*np.array([1,-1])
        clusters_params['s'].append((center_point+1*ori) / img.shape[:2])
        clusters_params['e'].append((center_point-1*ori) / img.shape[:2])
        clusters_params['bp1'].append(intersec_points[0] / img.shape[:2])
        clusters_params['bp2'].append(intersec_points[1] / img.shape[:2])
        clusters_params['width'].append(width)
        
        clusters_params['color_rgb'].append(np.mean(img[cluster_mask], axis=0))
        
        clusters_params['center'].append(center_point/ img.shape[:2])
        clusters_params['num_pixels'].append(np.sum(cluster_mask))
        clusters_params['stddev'].append(np.mean(np.std(img[cluster_mask], axis=0)))
        
    for key in clusters_params.keys():
        clusters_params[key] = np.array(clusters_params[key])
    
    N = clusters_params['center'].shape[0]
    
    stddev = clusters_params['stddev']
    rel_num_pixels = 5 * clusters_params['num_pixels'] / np.sqrt(H * W)

    location = clusters_params['center']
    num_pixels_per_cluster = clusters_params['num_pixels'].reshape(-1, 1)
    s = clusters_params['s']
    e = clusters_params['e']
    cluster_width = clusters_params['width']
    
    location[..., 0] *= H
    location[..., 1] *= W
    s[..., 0] *= H
    s[..., 1] *= W
    e[..., 0] *= H
    e[..., 1] *= W

    s -= location
    e -= location
    
    color = clusters_params['color_rgb']

    c = (s + e) / 2. + np.stack([np.random.uniform(low=-1, high=1, size=[N]),                                                                 
                                 np.random.uniform(low=-1, high=1, size=[N])],             
                                 axis=-1)
    
    sec_center = (s + e + c) / 3.
    s -= sec_center                                                                                                          
    e -= sec_center                                                                                                           
    c -= sec_center
    
    rel_num_pix_quant = np.quantile(rel_num_pixels, q=[0.3, 0.99])
    width_quant = np.quantile(cluster_width, q=[0.3, 0.99])
    rel_num_pixels = np.clip(rel_num_pixels, rel_num_pix_quant[0], rel_num_pix_quant[1])
    cluster_width = np.clip(cluster_width, width_quant[0], width_quant[1])
    width = width_scale * rel_num_pixels.reshape(-1, 1) * cluster_width.reshape(-1, 1)
    s, e, c = [x * sec_scale for x in [s, e, c]]
    
    location, s, e, c, width, color = [x.astype(np.float32) for x in [location, s, e, c, width, color]]
    print("Num clusters: {}; Num strokes: {}".format(num_clusters,N))
    return location, s, e, c, width, color


def initialize_brushstrokes(content_img, num_strokes, canvas_height, canvas_width, sec_scale, width_scale, init='sp',init_prob = None,offset=0.5,init_width=None):

    if init == 'random':
        # Brushstroke colors
        color = np.random.rand(num_strokes, 3)

        # Brushstroke widths
        width = np.random.rand(num_strokes, 1) * width_scale
        
        # Brushstroke locations
        location = np.stack([np.random.rand(num_strokes) * canvas_height, np.random.rand(num_strokes) * canvas_width], axis=-1)

        # Start point for the Bezier curves
        s = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # End point for the Bezier curves
        e = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # Control point for the Bezier curves
        c = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # Normalize control points
        sec_center = (s + e + c) / 3.0
        s, e, c = [x - sec_center for x in [s, e, c]]
        s, e, c = [x * sec_scale for x in [s, e, c]]
    elif init=="vor":
        flat_prob = init_prob.flatten()
        sample_index = np.random.choice(a=flat_prob.size, p=flat_prob,size=(num_strokes,))
        unique_samples = np.unique(sample_index)
        adjusted_index = np.unravel_index(sample_index, init_prob.shape)
        
        indices = np.stack(adjusted_index,-1)
        vor = scipy.spatial.Voronoi(indices)
        
        image_points = np.reshape(np.stack(np.meshgrid(np.arange(init_prob.shape[1]),np.arange(init_prob.shape[0])),-1),(-1,2))
        ip2 = image_points**2
        sp2 = indices**2

        cluster_image = np.zeros(init_prob.shape)
        for i in range(image_points.shape[0]):
          dists = np.sum(ip2[i,:]-2*image_points[i:i+1,:]*indices+sp2,-1)
          cluster_image[image_points[i,1],image_points[i,0]] = np.argmin(dists)
        cluster_image = cluster_image.astype(np.int)
        location, s, e, c, width, color = clusters_to_strokes(cluster_image,
                                                              content_img,
                                                              canvas_height,
                                                              canvas_width,
                                                              sec_scale=sec_scale,
                                                              width_scale=width_scale)
    else:
        if init_prob is not None:
            sorted_vals = np.sort(init_prob.flatten())
            norm_cdf = scipy.stats.norm.cdf(sorted_vals)
            norm_cdf = norm_cdf-np.min(norm_cdf)
            norm_cdf = norm_cdf/np.max(norm_cdf)
            err_thres = sorted_vals[np.abs(norm_cdf-offset).argmin()]
            segments = slic(content_img,
                            n_segments=num_strokes,
                            min_size_factor=0.02,
                            max_size_factor=4.,
                            compactness=2,
                            sigma=1,
                            start_label=0,
                            mask=init_prob>=err_thres)
            
        else:
            segments = slic(content_img,
                            n_segments=num_strokes,
                            min_size_factor=0.02,
                            max_size_factor=4.,
                            compactness=2,
                            sigma=1,
                            start_label=0)
        plt.imshow(segmentation.mark_boundaries(content_img, segments))
        plt.show()
        location, s, e, c, width, color = clusters_to_strokes(segments,
                                                              content_img,
                                                              canvas_height,
                                                              canvas_width,
                                                              sec_scale=sec_scale,
                                                              width_scale=width_scale,
                                                              init_prob=init_prob,
                                                              offset=offset,
                                                              init_width=init_width)

    return location, s, e, c, width, color
