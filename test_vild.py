import matplotlib.pyplot as plt
from PIL import Image
from reward.detector import VILD

image_path = "reward/tmp/color.png"
image = Image.open(image_path)
plt.imshow(image)
vild = VILD()
vild.vild_detect(image_path,["blocks","bowl"],verbose=True)