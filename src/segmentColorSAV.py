#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg      import Image
from cv_bridge                  import CvBridge
from skimage.color            import rgb2hsv
from scipy.misc                 import bytescale
from message_filters          import ApproximateTimeSynchronizer, Subscriber

from dynamic_reconfigure.server import Server
from baxter_perception.cfg          import baxter_perceptionConfig

nodename        = "imageTest"
imgTopic        = "/kinect2/qhd/image_color_rect"
depthTopic      = "/kinect2/qhd/image_depth_rect"
outEdgesTopic   = "/edges2"

# temp
fRgbImg = None
fDepImg = None
fEdges  = None

class edgeFilter:
    def __init__(self, nodename,imgTopic,depthTopic,outEdgesTopic,queue_size=100,slop=1.0):
        rospy.init_node(nodename)
        self.bridge = CvBridge()
        self.edgePub = rospy.Publisher(outEdgesTopic,Image,queue_size=100)
        self.coloPub = rospy.Publisher("colorim",Image,queue_size=100)
        self.depzPub = rospy.Publisher("depthim",Image,queue_size=100)
        self.tss =  ApproximateTimeSynchronizer( [Subscriber(imgTopic,Image),
                                                  Subscriber(depthTopic,Image)],
                                                  queue_size,
                                                  slop,
                                               )
        self.tss.registerCallback(self.gotRGBDimage)    
        
        self.HueH = 0
        self.HueL = 0
        self.Sat    = 0
        self.Val    = 0
        self.DistMin = 0
        self.DistMax = 0

        self.srv = Server(baxter_perceptionConfig, self.callback)



    def callback(self, config, level):
        self.HueH = config["HueH"]
        self.HueL = config["HueL"]
        self.Sat    = config["Sat"]
        self.Val    = config["Val"]
        self.DistMin = config["DistMin"]
        self.DistMax = config["DistMax"]
        rospy.logdebug("""Reconfigure Request: {HueH}, {HueL},\ 
              {Sat}, {Val}""".format(**config))
        return config

    def gotRGBDimage(self, rgb, depth):
        global fRgbImg  # temp testvar
        global fDepImg  # temp testvar
        global fEdges   # temp testvar

        imrgb = self.bridge.imgmsg_to_cv2(rgb)              # ros msg to img
        fRgbImg = imrgb                                                 # temp testvar
        imHSV = rgb2hsv(imrgb[...,[2,1,0]])                    # BGR to RGB to HSV
        
        maskH = np.logical_and( (imHSV[:,:,0] > self.HueL) , (imHSV[:,:,0] < self.HueH) ) 
        maskS = imHSV[:,:,1] > self.Sat
        maskV = imHSV[:,:,2] > self.Val
        mask = np.multiply(maskH,maskS,maskV)

        imdepth   = self.bridge.imgmsg_to_cv2(depth)        # ros msg to img
        fDepImg = imdepth  # temp testvar
        depthMask = bytescale(imdepth,self.DistMin,self.DistMax)#[:,300:600]  # bytescale and crop
        depthMask[depthMask==depthMask.max()]=depthMask.min()
        depthMask = depthMask > 1                           # binarize
        #depthMask = binary_fill_holes(depthMask.astype(np.uint8),structure=np.ones((1,4))).astype(np.uint8) 
    
        #res = cv2.bitwise_and(bw,bw,mask=depthMask.astype(np.uint8) ) 
        res = np.logical_and(depthMask,mask)
        fEdges = res  # temp testbar

        self.edgePub.publish(self.bridge.cv2_to_imgmsg(bytescale(res)))
        self.coloPub.publish(rgb)
        self.depzPub.publish(depth)


if __name__ == '__main__':
    edgeFilter(nodename,imgTopic,depthTopic,outEdgesTopic)
    while not rospy.is_shutdown():
        rospy.spin()
