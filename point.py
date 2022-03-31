import numpy as np 

class Point(object):
    def __init__(self, x_max, x_min, y_max, y_min):
        self.x = 50
        self.y = 50
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max)
        self.y = self.clamp(y, self.y_min, self.y_max)
    
    def get_position(self):
        return (self.x, self.y)