# -*- coding: utf-8 -*-

# 1. 对 bbox 进行顺时针排序
# 寻找一系列点的四个角(左上、右下、左下、右上)
def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 2. 对一系列的点进行顺时针排序
def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts



# 3. 图像细化操作
# - 方式 1: GaussianBlu + canny edge detection -> 转换为边缘
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
edge = cv2.Canny(img_gray, 50, 7, 3)

# - 方式 2: erode op -> 腐蚀操作: 去着重配置 kernel 和 iterations
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img_gray, kernel, iterations=1)

# - 方式 3: skeleton
ret, binary = cv2.threshold(img_gray, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binary[binary == 255] = 1
erosion = morphology.skeletonize(binary)
skeleton = erosion.astype(np.uint8) * 255

# 4. 图像去除低于阈值的连通域
thresh = ...
contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < thresh:
        cv2.drawContours(img_gray, [contours[i]], -1, (0, 255, 0), thickness=-1)
            continue
            
 # 5. 用单卡加载分布式训练的模型， 需要移除对应的 state 中的 module
#  可以调用这个模型， 更新一下 state _dict 就好了
def remove_module_dict(state_dict, is_print=False):
    if is_print: print(state_dict.keys())
    new_state_dict = {
        k.replace('module.',''): v for k, v in 
        state_dict.items()
    }
    if is_print: print(new_state_dict.keys())
    return new_state_dict


# 6. 求两个直线的交点
def line_intersection(line1, line2):
    """
        line1: (x1, y1, x2, y2),  line2: (x3, y3, x4, y4)
        reference:
            https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def _det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = _det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (_det(*line1), _det(*line2))
    x = _det(d, xdiff) / div
    y = _det(d, ydiff) / div
    return x, y



