import numpy as np 
import cv2 
import random
import time
import math
from point import Point

NUM_ACTIONS = 8
SCALE = 5
class Robot(Point):
    def __init__(self, x_max, x_min, y_max, y_min):
        super(Robot, self).__init__(x_max, x_min, y_max, y_min)

        self.icon = cv2.imread('robot.png')
        self.icon_h = 16
        self.icon_w = 16
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        self.u = 0
        self.w = 0
        self.theta =  0
        self.a = 0

    def move(self, action):
        direct = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.x += direct[action][0] * SCALE
        self.y += direct[action][1] * SCALE

        
    
        
        
