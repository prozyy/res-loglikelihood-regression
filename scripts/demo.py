"""Demo script."""
import torch
import torch.multiprocessing as mp
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/code/res-loglikelihood-regression")
from rlepose.models import builder
# from rlepose.opt import cfg, opt
# from rlepose.trainer import validate, validate_gt
# from rlepose.utils.env import init_dist
# from rlepose.utils.transforms import get_coord
skeleton = ( (0,21),(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7,9), (9,22), (6,8), (8,10), (10,23), (5,11), (11,12),(11,13), (13, 15), (15, 17), (15, 19) ,(6,12),(12,14),(14,16),(16,18),(16,20) )

def vis_keypoints(img, kps, kps_lines = skeleton, kp_thresh=0.3, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=5, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=5, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

class RLE:
    def __init__(self):
        MODEL = {'TYPE': 'RegressFlow', 'PRETRAINED': '', 'TRY_LOAD': '', 'NUM_FC_FILTERS': [-1], 'HIDDEN_LIST': -1, 'NUM_LAYERS': 50}
        DATA_PRESET = {'TYPE': 'simple', 'SIGMA': 2, 'NUM_JOINTS': 17, 'IMAGE_SIZE': [256, 192], 'HEATMAP_SIZE': [64, 48]}
        self.m = builder.build_sppe(MODEL, preset_cfg=DATA_PRESET)
        self.m.load_state_dict(torch.load("/code/res-loglikelihood-regression/coco-laplace-rle.pth", map_location='cpu'), strict=True)
        self.m = self.m.cuda().eval()

    def extract_keypoints(self,image,bbox=None):
        image = image[:,:,::-1].copy()
        if bbox is not None:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cw = (x2 - x1 + 1) * 1.25
            ch = (y2 - y1 + 1) * 1.25

            x1 = cx - cw / 2
            x1 = int(np.max([x1,0]))
            x2 = cx + cw / 2
            x2 = int(np.min([x2,image.shape[1]]))
            y1 = cy - ch / 2
            y1 = int(np.max([y1,0]))
            y2 = cy + ch / 2
            y2 = int(np.min([y2,image.shape[1]]))

            image = image[y1:y2,x1:x2,:]
        else:
            x1 = 0
            y1 = 0
        h,w,_ = image.shape

        image_resized = cv2.resize(image,(192,256))
        image_input = image_resized.transpose((2,0,1)).astype(np.float32).reshape(1,3,256,192)

        image_input[0,0,:,:] = image_input[0,0,:,:] / 255 -0.406
        image_input[0,1,:,:] = image_input[0,1,:,:] / 255 -0.457
        image_input[0,2,:,:] = image_input[0,2,:,:] / 255 -0.480

        input = torch.from_numpy(image_input).cuda()
        out = self.m(input)

        kpts = out["pred_jts"].detach().cpu().numpy()
        score = out["maxvals"].detach().cpu().numpy()

        kpts[:,:,0] = (kpts[:,:,0] + 0.5 ) * w + x1
        kpts[:,:,1] = (kpts[:,:,1] + 0.5 ) * h + y1

        kpts_ = np.concatenate([kpts,score],axis=2)[0].transpose()

        kpts_24 = np.zeros((3,24))
        kpts_24[:,:17] = kpts_

        return kpts_24.transpose()[np.newaxis,...]

if __name__ == "__main__":

    model = RLE()
    image = cv2.imread("test.jpg")
    kpts_24 = model.extract_keypoints(image)
    image_disp = vis_keypoints(image,kpts_24[0].transpose())
    cv2.imwrite("result.jpg",image_disp)
