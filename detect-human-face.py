import cv2
import time
from argparse import ArgumentParser
from matplotlib import pyplot as plt

def build_argparser():
  parser = ArgumentParser()
  general = parser.add_argument_group('General')
  general.add_argument('-i', '--input', required=True,
                          help=" Path to the input image ")
  general.add_argument('-conf', '--confidence', type = float, default=0.2,
                          help=" Confidence threshold for detection ")
  general.add_argument('-iou', '--nms_threshold', type = float, default=0.4,
                          help=" Non max suppression threshold ")
  general.add_argument('-cfg', '--config', default="/content/drive/My Drive/darknet/cfg/yolo-obj.cfg",
                          help=" PATH to configuration file ")
  general.add_argument('-w', '--weights', default="/content/drive/My Drive/darknet/backup/yolo-obj_last.weights",
                          help=" PATH to weights file ")
  return parser

args = build_argparser().parse_args()

CONFIDENCE_THRESHOLD = args.confidence
NMS_THRESHOLD = args.nms_threshold
yolo_config = args.config
yolo_weights = args.weights
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = ['person', 'head']

frame = cv2.imread(args.input)

net = cv2.dnn.readNet(yolo_weights, yolo_config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

start = time.time()
classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
end = time.time()


for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %f" % (class_names[classid[0]], score)
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite("prediction.jpg", frame)
# plt.imshow(frame)
# plt.show()