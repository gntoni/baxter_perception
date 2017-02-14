#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import measure, img_as_float
from scipy.ndimage import convolve
from cv_bridge import CvBridge
from threading import Thread, Lock
from sensor_msgs.msg import Image
from baxter_perception.srv import graspPoint, graspPointRequest, graspPointResponse
from edgeDetector.network import edgeDetectCNN
from edgeDetector.dataset_generation.patchGenerator import unlabeled_patching

NODENAME = "getGraspPoint"
EDGES_TOPIC = "/edges"

HPREWITT_WEIGHTS = np.array([
                                    [1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]]) / 3.0

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

PATCH_SIZE = (50, 50)

MODEL = "/home/toni/Data/bagsTowels/model_350_epoch.npy"  # TODO get from param server


class Cget_grasp_service(object):
    """
    Class to provide a ROS service that given a depth image and it's edges,
    returns a grasping point for the best position.
    """
    def __init__(self):
        print "[get_grasp_point_srv]  creating server..."
        self.bridge = CvBridge()
        self.net = edgeDetectCNN(PATCH_SIZE[0], PATCH_SIZE[1])
        model = np.load(MODEL)
        self.net.set_network_model(model)
        print "[get_grasp_point_srv]  server ready!"

    def handle_get_grasp_point(self, req):
        print "[get_grasp_point_srv] Grasp point calculation request received."
        edge_image = self.bridge.imgmsg_to_cv2(req.edges)  # ros msg to img
        depth_image = self.bridge.imgmsg_to_cv2(req.depth)  # ros msg to img
        imH, imW = depth_image.shape

        p = unlabeled_patching(
                                        edge_image.reshape((1, imH, imW)),
                                        depth_image.reshape((1, imH, imW))
                                        )
        pred = self.net.test(np.array(p).reshape((-1, 1, PATCH_SIZE[0], PATCH_SIZE[1])))
        pred = np.argmax(pred, axis=1)

        pred_image = np.zeros_like(edge_image)
        print "[get_grasp_point_srv] recovering image"
        index = 0
        for i in range(PATCH_SIZE[0], imH - PATCH_SIZE[0]):
            for j in range(PATCH_SIZE[1], imW - PATCH_SIZE[1]):
                if edge_image[i, j] == 255:
                    pred_image[i, j] = pred[index]
                    index += 1
        pred_image *= 255

        # Find contours of the image
        contours = measure.find_contours(pred_image, 0.8)

        # Find vertical derivative (horizontal edges)
        prewitt_im = convolve(img_as_float(pred_image), HPREWITT_WEIGHTS)

        horizontality = []
        indices = []
        print "len indices: " + str(len(indices))
        for index, contour in enumerate(contours):
            if cv2.arcLength(contour.astype("float32"), False) > 70:
                print "contour: " + str(cv2.arcLength(contour.astype("float32"), False))
                a = prewitt_im[contour[:, 0].astype(int), contour[:, 1].astype(int)]
                b = sum(a) / float(len(a))
                horizontality.append(b)
                indices.append(index)

        if len(indices) == 0:
            return graspPointResponse()
        horizontality = np.array(horizontality).astype(float)
        indices = np.array(indices).astype(float)
        args = np.argsort(horizontality)
        cd_points = []
        for i in range(1):
            contour_index = int(indices[args][i])
            midp = int(len(contours[contour_index])/2)
            cd_points.append((int(contours[contour_index][midp, 1]), int(contours[contour_index][midp, 0])))

        outputMsg = graspPointResponse()
        outputMsg.graspPoint.x, outputMsg.graspPoint.y = cd_points[0]
        outputMsg.graspPoint.z = 0

        np.save("ct", contours[contour_index])
        return outputMsg


def get_grasp_point_server():
    rospy.init_node('get_grasp_point_server')
    print "get_grasp_point service initialized. Awaiting input images."
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('get_grasp_point_server')
    ggpt_srv = Cget_grasp_service()
    p = rospy.Service('get_grasp_point', graspPoint, ggpt_srv.handle_get_grasp_point)
    get_grasp_point_server()
