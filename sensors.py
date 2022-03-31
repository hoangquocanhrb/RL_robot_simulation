
import math
import numpy as np 

NUM_LASERS = 50
LASER_RANGE = 500

def uncertainty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(sigma**2)
    distance, angle = np.random.multivariate_normal(mean, covariance)
    distance = max(distance, 0)
    # angle = max(angle, 0)
    # return [distance, angle]
    return distance

class LaserSensor:
    def __init__(self, map, uncertainty, x_max, x_min, y_max, y_min):
        self.Range = LASER_RANGE
        self.speed = 4
        self.sigma = np.array([uncertainty[0], uncertainty[1]])
        self.position=[0,0]
        self.num_lasers = NUM_LASERS
        self.map = map
        self.w = self.map.shape[1]
        self.h = self.map.shape[0]
        self.sensedObstacles = []

        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
    
    def distance(self, obstaclePosition):
        px = (obstaclePosition[0] - self.position[0])**2
        py = (obstaclePosition[1] - self.position[1])**2
        return math.sqrt(px+py)

    def set_position(self, x, y):
        self.position[1] = x
        self.position[0] = y 

    def get_position(self):
        return self.position

    def sense_obstacle(self):
        data = []
        self.sensedObstacles = []
        x1, y1 = self.position[0], self.position[1]
        
        for angle in np.linspace(0, 2*math.pi, self.num_lasers, False):
            x2, y2 = (x1 + self.Range * math.cos(angle), y1 - self.Range * math.sin(angle))
            y = y2
            x = x2
            
            for i in range(0,200):
                u = i/200
                x = int(x2*u + x1*(1-u))
                y = int(y2*u + y1*(1-u))
                if 0<x<self.w and 0<y<self.h:
                    color = self.map[y][x]
                    
                    if (color[0] == 0 and color[1] == 0 and color[2]==0) or i==199:
                        distance = self.distance((x,y))
                        # output = uncertainty_add(distance, angle, self.sigma)
                        output = distance/30
                        # output.append(self.position)

                        data.append(output)
                        break
                    
            x = max(min(x, self.x_max), self.x_min)
            y = max(min(y, self.y_max), self.y_min)
            self.sensedObstacles.append([x, y])
            
        return data

    def obs_pose(self):
        
        return self.sensedObstacles