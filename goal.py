import numpy as np 
from point import Point
import cv2 

class Goal(Point):
    def __init__(self, x_max, x_min, y_max, y_min):
        super(Goal, self).__init__(x_max, x_min, y_max, y_min)
        self.icon = cv2.imread('apple.jpeg')
        self.icon_h = 16
        self.icon_w = 16
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))