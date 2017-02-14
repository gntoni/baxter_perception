#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from baxter_perception.cfg import baxter_perceptionConfig

def callback(config, level):
    rospy.logdebug("""Reconfigure Request: {HueH}, {HueL},\ 
          {Sat}, {Val}""".format(**config))
    return config

if __name__ == "__main__":
    rospy.init_node("baxter_perception", anonymous = True)

    srv = Server(baxter_perceptionConfig, callback)
    rospy.spin()

