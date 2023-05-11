import cv2
import numpy as np
import onnxruntime as ort
import time

def get_input_feed(img_tensor):
    input_feed = {}
    input_name = []
    for node in sess.get_inputs():
        input_name.append(node.name)
    for name in input_name:
        input_feed[name] = img_tensor
    return input_feed

def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep

def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #   删除为1的维度
    #	删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    # -------------------------------------------------------
    #	通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #	1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------
    output = []

    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output

def draw(image, box_data):
    # -------------------------------------------------------
    #	取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

# 加载模型
sess = ort.InferenceSession("best-sim.onnx")
# 定义类别名称
CLASSES = ['nlb']
# 打开摄像头
cap = cv2.VideoCapture('nlb.mp4')
width = 1080 #定义摄像头获取图像宽度
height = 1080 #定义摄像头获取图像长度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) #设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) #设置长度
# 定义输出尺寸
# 定义阈值
conf_threshold = 0.65
nms_threshold = 0.65

# 初始化FPS计数器
#fps_counter = 0


while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        break
    # 调整尺寸
    resized_frame = cv2.resize(frame, (320, 320))
    start_time = time.time()
    # 转换颜色空间
    input_frame = resized_frame[:, :, ::-1].transpose(2, 0, 1)
    # 转换为浮点数类型
    input_frame = input_frame.astype(dtype=np.float32)
    # 归一化
    input_frame /= 255.0
    # 增加一个维度
    input_frame = np.expand_dims(input_frame, axis=0)

    # 进行推理
    input_feed = get_input_feed(input_frame)
    outputs = sess.run(None, input_feed)[0]
    # 解析输出
    outbox = filter_box(outputs, conf_threshold, nms_threshold )
    if len(outbox) != 0:
        # 绘制框和标签
        draw(resized_frame, outbox)
    # 显示FPS
    fps_counter = 1
    elapsed_time = time.time() - start_time
    fps = (fps_counter / elapsed_time)/5
    print('fps',fps)
    cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    # 显示图像
    cv2.imshow('frame', resized_frame)
    # 按q键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头
cap.release()
cv2.destroyAllWindows()