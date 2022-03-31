import numpy as np
from robot import Robot, NUM_ACTIONS
from goal import Goal
import cv2 
import time 
import math
from PIL import Image
from sensors import LaserSensor, NUM_LASERS, LASER_RANGE
import random 

NUM_INPUTS = NUM_LASERS + 1 + 12#distance and angle to goal + orient
MAX_DISTANCE = LASER_RANGE
NUM_OUTPUTS = NUM_ACTIONS

class RoomMap():
    def __init__(self, map_path):

        self.map = cv2.imread(map_path)
        
        self.observation_shape = self.map.shape
        
        self.y_min = int(self.observation_shape[0]*0.03)
        self.x_min = int(self.observation_shape[1]*0.02)
        self.y_max = int(self.observation_shape[0]*0.97)
        self.x_max = int(self.observation_shape[1]*0.98)
                
        self.robot = Robot(self.x_max, self.x_min, self.y_max, self.y_min)
        self.goal = Goal(self.x_max, self.x_min, self.y_max, self.y_min)
        self.goal.set_position(50, 600) #temp
        
        self.laser = LaserSensor(self.map, uncertainty=(0.5, 0.01), 
            x_max=self.x_max, x_min=self.x_min, y_max=self.y_max, y_min=self.y_min)

        self.pointClouds = self.laser.obs_pose()
        self.lidar = []
        self.orientation = []
        self.data = []

    def draw_on_canvas(self):
        self.canvas = self.map.copy()
        robot_shape = self.robot.icon.shape
        goal_shape = self.goal.icon.shape
        x_rb = robot_shape[0]//2
        y_rb = robot_shape[1]//2

        x_g = goal_shape[0]//2
        y_g = goal_shape[1]//2
        # self.show_sensorData()
        
        # self.laser.set_position(self.robot.y, self.robot.x)
        
        self.canvas[int(self.robot.y) - y_rb : int(self.robot.y) + y_rb,
                    int(self.robot.x) - x_rb : int(self.robot.x) + x_rb] = self.robot.icon
        self.canvas[int(self.goal.y) - y_g : int(self.goal.y) + y_g,
                    int(self.goal.x) - x_g : int(self.goal.x) + x_g] = self.goal.icon
        
        # self.robot_frame()
        
        for i in range(len(self.pointClouds)):
            cv2.line(self.canvas, (int(self.robot.x), int(self.robot.y)), (self.pointClouds[i][0], self.pointClouds[i][1]), (0,255,0), 1)
        cv2.imshow('MAP', self.canvas)
    
    def robot_frame(self):
        n = 40
        centerx, centery = (self.robot.x, self.robot.y)
        x_axis = (int(centerx + n*math.cos(self.robot.theta)), int(centery + n*math.sin(self.robot.theta)))
        y_axis = (int(centerx + n*math.cos(self.robot.theta + math.pi/2)),
                int(centery + n*math.sin(self.robot.theta + math.pi/2)))
        
        cv2.line(self.canvas, (int(self.robot.x), int(self.robot.y)), x_axis, (255,255,0), 3)
        # cv2.line(self.canvas, (int(self.robot.x), int(self.robot.y)), y_axis, (255,0,0), 3)

    def show_sensorData(self):
        for point in self.pointClouds:
            self.map[point[1]][point[0]] = (0, 0, 255)

    def has_collided(self):
        for distance in self.lidar:
            if(distance < 0.5):
                return True
        return False

    def Robot2Goal(self):
        px = (self.robot.x - self.goal.x)**2
        py = (self.robot.y - self.goal.y)**2 
        distance = math.sqrt(px+py)
        distance /= 100
        ay = self.goal.y - self.robot.y
        ax = self.goal.x - self.robot.x 
        
        temp = 0 #angle >= 0 
        angle = math.atan2(ay, ax)
        
        # angle -= self.robot.theta

        # angle *= 10
        # if angle < 0:
        #     angle = -angle
        #     temp = 1
        # print(distance)
        
        return distance, angle
    
    def orient(self, angle):
        pos = angle/(math.pi/6)
        orientation = [0]*12
        pos = int(pos)
        pos = max(-12, min(11, pos))
        orientation[pos] = 10
        return orientation
    
        
    def random_pose(self):
        x = random.randint(self.x_min, self.x_max)
        y = random.randint(self.y_min, self.y_max)
        return [x, y]

    def reset(self):
        self.data = []
        
        self.robot = Robot(self.x_max, self.x_min, self.y_max, self.y_min)
        self.goal = Goal(self.x_max, self.x_min, self.y_max, self.y_min)
        self.laser = LaserSensor(self.map, uncertainty=(0.5, 0.01), 
            x_max=self.x_max, x_min=self.x_min, y_max=self.y_max, y_min=self.y_min)

        g_pose = [(400, 200), (300, 400), (500, 450), (520, 70), (100, 100), (100, 300), (100, 400), (400, 50), (250, 50)]
        # g_pose = [(550, 250), (350, 250), (400,350), (350, 120), (150,220)]
        # r_pose = [(50, 50), (60, 450)]
        # j = random.randint(0, 1)
        i = random.randint(0, len(g_pose)-1)
        # i = 8
        self.goal.set_position(g_pose[i][0], g_pose[i][1])

        # self.robot.set_position(r_pose[j][0], r_pose[j][1])
        # self.robot.theta = math.pi/2
        # self.laser.set_position(self.robot.y, self.robot.x)
        # self.lidar = self.laser.sense_obstacle()
        
        while(True):
            r_pose = self.random_pose()
            self.robot.set_position(r_pose[0], r_pose[1])
            self.laser.set_position(self.robot.y, self.robot.x)

            self.lidar = self.laser.sense_obstacle()

            distance, _= self.Robot2Goal()
            if(distance < 0.1):
                continue
            if(self.has_collided()):
                continue
            break
        
        self.pointClouds = self.laser.obs_pose()

        self.reward = 0.0
        self.fuel = 500

        self.done = False
        
        
        # for i in range(len(self.data[:-2])):
        #     self.data[i] /= 10
        distance, angle = self.Robot2Goal()

        orientation = self.orient(angle) # scale angle
        self.data = self.lidar.copy()
        self.data += orientation
        self.data.append(distance)
        # self.data.append(angle)
        return self.data
    
    def step(self, action):
        past_dis, past_angle= self.Robot2Goal()       
        self.robot.move(action)

        self.laser.set_position(self.robot.y, self.robot.x)
        self.lidar = self.laser.sense_obstacle()
        
        distance, angle= self.Robot2Goal()
        # print(distance)
        orientation = self.orient(angle)
        self.data = self.lidar.copy()
        self.data += orientation
        self.data.append(distance)
        # self.data.append(angle)

        # print(angle*90/math.pi)
        # print(orientation)

        self.pointClouds = self.laser.obs_pose()
        # for i in range(len(self.pointClouds)):
        #     self.pointClouds[i][0] = max(min(self.pointClouds[i][0], self.x_max), self.x_min)
        #     self.pointClouds[i][1] = max(min(self.pointClouds[i][1], self.y_max), self.y_min)
        # if action == 0:
        #     reward = -0.01
        # else:
        #     reward = 0.01
        reward = 0.0
        self.fuel -= 1
        out = False

        # print(distance)
        if self.has_collided() == True:
            print("--Collided--")
            self.done = True
            reward = -8
        
        if self.fuel == 0:
            print("--Out of fuel--")
            out = True
            self.done = True
            reward = -10
    
        if distance < 0.1:
            print("GOALLLLLLL")
            self.done = True
            reward = 20
        
        if distance < 1 and distance >= 0.1:
            if distance < past_dis:
              reward += 0.005
              
        '''
        elif distance < 0.5 and distance >= 0.1 :
            if distance < past_dis:
              # print('Near goal in range (10, 50)')
              reward += 0.03
              if angle <= past_angle:
                # print('Smaller angle (10, 50)')
                reward += 0.01
        '''
        return self.data, reward, self.done


    def render(self):

        self.draw_on_canvas()     
        cv2.waitKey(10)

# map = RoomMap('big_map.png')
# dt = 0
# lasttime = time.time()
# map.reset()

# for i in range(1000):
#     dt = (time.time() - lasttime)
#     lasttime = time.time()
    
#     obs, r, d = map.step(0, dt)
#     print(obs)
#     map.draw_on_canvas()
    
