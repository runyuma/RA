from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward,SimplifyPickEnvWithoutLangReward
from environments.environment import PickPlaceEnv,BOUNDS,PIXEL_SIZE
from environments.dummy_environment import dummypick
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
# from reward.detector import VILD
import cv2
if __name__ == "__main__":
  env = PickPlaceEnv(PutBlockInBowl)
  env.reset()
  height_map,_,_ = env.get_depth(BOUNDS,PIXEL_SIZE)
  image = env.get_image()
  height_map = height_map[:,:,np.newaxis]
  height_map = np.concatenate([height_map,height_map,height_map],axis=2)*255
  height_map = height_map.astype(np.uint8)
  print(image.shape,image.dtype,image.max(),image.min())
  print(height_map.shape,height_map.dtype,height_map.max(),height_map.min())
  cv2.imshow("color",image)
  cv2.imshow("height_map",height_map*20)
  cv2.waitKey(0)
  cv2.destroyAllWindows()