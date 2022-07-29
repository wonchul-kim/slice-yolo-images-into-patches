import os
import os.path as osp 
import cv2
import time
import numpy as np
import skimage.io
from glob import glob
import copy
from utils import xyxy2xywh, xywh2xyxy 

def slice_img_w_ann(image_path, out_name, out_dir_images, 
             boxes=[], yolo_classes=[], out_dir_labels=None, 
             mask_path=None, out_dir_masks=None,
             sliceHeight=416, sliceWidth=416,
             overlap=0.1, slice_sep='|', pad=0,
             skip_highly_overlapped_tiles=False,
             overwrite=False,
             out_ext='.png', verbose=False, vis=False):
    """
    Slice a large image into smaller windows, and also bin boxes
    Adapted from:
         https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/slice_im.py
    """
    if vis and not osp.exists(osp.join(out_dir_images, "vis")):
        os.makedirs(osp.join(out_dir_images, "vis"))

    if len(out_ext) == 0:
        im_ext = '.' + image_path.split('.')[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    image = skimage.io.imread(image_path)  #, as_grey=False).astype(np.uint8)  # [::-1]
    print("image.shape:", image.shape)
    if mask_path:
        mask = skimage.io.imread(mask_path)
    win_h, win_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)
    
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            out_boxes_yolo = []
            out_classes_yolo = []
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image.shape[0]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight
            # print(">> xmin, xmax, ymin, ymax = ", xmin, xmax, ymin, ymax)
            vis_bboxes = []
            vis_labels = []
            # find boxes that lie entirely within the window
            if len(boxes) > 0:
                out_path_label = os.path.join(
                    out_dir_labels,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + '.txt')
                for j,b in enumerate(boxes):
                    yolo_class = yolo_classes[j]
                    xb0, yb0, xb1, yb1 = b
                    # print("------------------------------------------------------------------------------")
                    out_box_tmp = []
                    if (xb0 >= xmin) and (yb0 >= ymin):
                        if (xb1 <= xmax) and (yb1 <= ymax):
                            out_box_tmp = [xb0 - xmin, yb0 - ymin,
                                        xb1 - xmin, yb1 - ymin] # x, y, x, y
                        elif (xb1 >= xmax >= xb0) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xb0 - xmin, yb0 - ymin,
                                        xmax - xmin, ymax - ymin] 
                        elif (xb1 <= xmax) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xb0 - xmin, yb0 - ymin,
                                        xb1 - xmin, ymax - ymin] 
                        elif (xb1 >= xmax >= xb0) and (yb1 <= ymax):
                            out_box_tmp = [xb0 - xmin, yb0 - ymin,
                                        xmax - xmin, yb1 - ymin]
                    elif (xb0 <= xmin <= xb1) and (yb0 >= ymin):      
                        if (xb1 <= xmax) and (yb1 <= ymax):
                            out_box_tmp = [xmin - xmin, yb0 - ymin,
                                        xb1 - xmin, yb1 - ymin]   
                        elif (xb1 <= xmax) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xmin - xmin, yb0 - ymin,
                                        xb1 - xmin, ymax - ymin] # x, y, x, y
                        elif (xb1 >= xmax >= xb0) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xmin - xmin, yb0 - ymin,
                                        xmax - xmin, ymax - ymin] 
                        elif (xb1 >= xmax >= xb0) and (yb1 <= ymax):
                            out_box_tmp = [xmin - xmin, yb0 - ymin,
                                        xmax - xmin, yb1 - ymin] 
                    elif (xb0 >= xmin) and (yb0 <= ymin <= yb1):                
                        if (xb1 >= xmax >= xb0) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xb0 - xmin, ymin - ymin,
                                        xmax - xmin, ymax - ymin] # x, y, x, y
                        elif (xb1 >= xmax >= xb0) and (yb1 <= ymax):
                            out_box_tmp = [xb0 - xmin, ymin - ymin,
                                        xmax - xmin, yb1 - ymin] 
                        elif (xb1 <= xmax) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xb0 - xmin, ymin - ymin,
                                        xb1 - xmin, ymax - ymin] # x, y, x, y
                        elif (xb1 <= xmax) and (yb1 <= ymax):
                            out_box_tmp = [xb0 - xmin, ymin - ymin,
                                        xb1 - xmin, yb1 - ymin] 
                    elif (xb0 <= xmin <= xb1) and (yb0 <= ymin <= yb1):                
                        if (xb1 >= xmax >= xb0) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xmin - xmin, ymin - ymin,
                                        xmax - xmin, ymax - ymin] # x, y, x, y
                        elif (xb1 <= xmax) and (yb1 >= ymax >= yb0):
                            out_box_tmp = [xmin - xmin, ymin - ymin,
                                        xb1 - xmin, ymax - ymin] 
                        elif (xb1 <= xmax) and (yb1 <= ymax):
                            out_box_tmp = [xmin - xmin, ymin - ymin,
                                        xb1 - xmin, yb1 - ymin] 
                        elif (xb1 >= xmax >= xb0) and (yb1 <= ymax):
                            out_box_tmp = [xmin - xmin, ymin - ymin,
                                        xmax - xmin, yb1 - ymin] 

                    if len(out_box_tmp) != 0:
                        for idx, tmp in enumerate(out_box_tmp):
                            if idx == 0 or idx == 2:
                                if tmp > sliceWidth:
                                    raise RuntimeError(f"Somethig wrong with the calculation of width")
                            if idx == 1 or idx == 3:
                                if tmp > sliceHeight:
                                    raise RuntimeError(f"Somethig wrong with the calculation of height")
                        # print("1. xyxy", out_box_tmp)
                        # print(xmin, xmax, ymin, ymax)
                        # print(xb0, yb0, xb1, yb1)
                        yolo_coords = xyxy2xywh((sliceHeight, sliceWidth), out_box_tmp)
                        # print("2. xywh:", yolo_coords)
                        out_boxes_yolo.append(yolo_coords)
                        out_classes_yolo.append(yolo_class)

                        vis_bboxes.append(out_box_tmp)
                        vis_labels.append(yolo_class)
            
                # skip if no labels?
                if len(out_boxes_yolo) == 0:
                    continue

                # save yolo labels
                txt_outfile = open(out_path_label, "w")     
                for yolo_class, yolo_coord in zip(out_classes_yolo, out_boxes_yolo):                          
                    outstring = str(yolo_class) + " " + " ".join([str(str(round(a, 3))) for a in yolo_coord]) + '\n'
                    if verbose: 
                         print("  outstring:", outstring.strip())
                    txt_outfile.write(outstring)
                txt_outfile.close()                

            # save mask, if desired
            if mask_path:
                mask_c = mask[y:y + sliceHeight, x:x + sliceWidth]
                outpath_mask = os.path.join(
                    out_dir_masks,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + im_ext)
                skimage.io.imsave(outpath_mask, mask_c, check_contrast=False)

            # extract image
            window_c = image[y:y + sliceHeight, x:x + sliceWidth]
            outpath = os.path.join(
                out_dir_images,
                out_name + slice_sep + str(y) + '_' + str(x) + '_'
                + str(sliceHeight) + '_' + str(sliceWidth)
                + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                + im_ext)
            if vis:
                vis_img = copy.deepcopy(window_c)
                vis_img = skimage.img_as_ubyte(vis_img)

                for (vis_bbox, vis_label) in zip(vis_bboxes, vis_labels):
                    print("...", vis_img.shape, vis_bbox)
                    cv2.rectangle(vis_img, (int(vis_bbox[0]), int(vis_bbox[1])), (int(vis_bbox[2]), int(vis_bbox[3])), (255, 0, 0), 2)
                cv2.imwrite(osp.join(out_dir_images, 'vis', out_name + slice_sep + str(y) + '_' + str(x) + '_'
                                    + str(sliceHeight) + '_' + str(sliceWidth) + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                            + im_ext), vis_img)
            if not os.path.exists(outpath):
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            elif overwrite:
                skimage.io.imsave(outpath, window_c, check_contrast=False)

                
            else:
                print("outpath {} exists, skipping".format(outpath))
                                                                                                 
    print("Num slices:", n_ims,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")

    return
    
if __name__ == '__main__':
    dataset_dir = "image_directory_path"
    output_dir = "output_path"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_files = glob(osp.join(dataset_dir, '*.png'))
    vis = False
    slice_height, slice_width = 640, 896
    overlap = 0.1
    slice_sep = '|'
    out_ext = '.png'
    verbose = False

    for img_file in img_files:
        img = cv2.imread(img_file)
        fn = osp.split(osp.splitext(img_file)[0])[-1]

        bboxes = [] # x0, y0, x1, y1
        labels = []


        f = open(osp.join(dataset_dir, fn + '.txt'))
        while True:
            line = f.readline()
            if not line: break

            annotation = list(map(float, line.split(" ")))

            labels.append(int(annotation[0]))
            bboxes.append(xywh2xyxy(img.shape, annotation[1:])) 

        f.close()

        vis_img = copy.deepcopy(img)
        vis_img = skimage.img_as_ubyte(vis_img)

        print(bboxes)
        for (vis_bbox, vis_label) in zip(bboxes, labels):
            cv2.rectangle(vis_img, (int(vis_bbox[0]), int(vis_bbox[1])), (int(vis_bbox[2]), int(vis_bbox[3])), (255, 0, 0), 3)
        cv2.imwrite(osp.join(output_dir, 'vis', fn + '.png'), vis_img)



        slice_img_w_ann(img_file, fn, output_dir, bboxes, labels, output_dir, sliceHeight=slice_height, sliceWidth=slice_width, overlap=overlap, vis=vis)  
       
       
