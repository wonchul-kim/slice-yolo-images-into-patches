def xyxy2xywh(imgsz, xyxy):

    if not len(imgsz) >= 2:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")
    elif imgsz[0] <= 3:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")

    if isinstance(xyxy, list):
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    else:
        raise RuntimeError(f"xyxy must be list such as [x1, y1, x2, y2]")

    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    
    dw = 1./imgsz[1]
    dh = 1./imgsz[0]
    
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    
    return [x, y, w, h]

def xywh2xyxy(imgsz, xywh):

    if not len(imgsz) >= 2:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")
    elif imgsz[0] <= 3:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")

    if isinstance(xywh, list):
        x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    else:
        raise RuntimeError(f"xywh must be list such as [x, y, w, h]")

    dh, dw = imgsz[0], imgsz[1]
    l = ((x - w / 2) * dw) # x0
    r = ((x + w / 2) * dw) # x1
    t = ((y - h / 2) * dh) # y0
    b = ((y + h / 2) * dh) # y1
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1   

    return [l, t, r, b] # x0, y0, x1, y1