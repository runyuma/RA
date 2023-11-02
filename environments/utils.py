import numpy as np
import cv2
def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False,pick = True,ee = "gripper"):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 1] - u * pixel_size

    if not skip_height:
        if ee == "gripper":
            if pick:
                z = bounds[2, 0] + height[u, v] - 0.02
            else:
                z = bounds[2, 0] + height[u, v] + 0.06
            z = np.clip(z,0.02,0.15)
        else:
            if pick:
                z = bounds[2, 0] + height[u, v] - 0.005
            else:
                z = bounds[2, 0] + height[u, v] + 0.07
            z = np.clip(z,0.005,0.20)
            z = z - 0.065# offset
    else:
        z = 0.0
    return [x, y, z]
def xyz_to_pix(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((-position[1] + bounds[1, 1]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return [u, v]

class mouse_demo():
    def __init__(self,img,hmap,bounds = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]]) ,pix_size = 0.00267857):
        self.img = img
        # self.img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.hmap = hmap
        self.pick_points = []
        self.place_points = []
        self.bounds = bounds
        self.pix_size = pix_size
    def pick_callback(self,event,u,v,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pick_points = [v,u]
            self.pick_points = np.clip(self.pick_points,10,230-1)
            print("**pick point**",self.pick_points)
    def place_callback(self,event,u,v,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.place_points = [v,u]
            self.place_points = np.clip(self.place_points,10,230-1)
            print("**place point**",self.place_points)
    def get_pick_point(self):
        cv2.namedWindow('pick')
        cv2.setMouseCallback('pick',self.pick_callback)
        while True:
            cv2.imshow('pick',self.img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or len(self.pick_points) == 2:
                break
        cv2.destroyAllWindows()
        return self.pick_points
    def get_place_point(self):
        cv2.namedWindow('place')
        cv2.setMouseCallback('place',self.place_callback)
        while True:
            cv2.imshow('place',self.img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or len(self.place_points) == 2:
                break
        cv2.destroyAllWindows()
        return self.place_points