# GelSight-homemade
A homemade version of MIT gelsight, with classic gradient reconstruction.

Gelsight is a visual tactile sensor used for surface reconstruction, which was introduced by MIT in their [CVPR paper](https://doi.org/10.1109/CVPR.2009.5206534). Our code is similar to [this repo](https://github.com/siyuandong16/gelsight_heightmap_reconstruction), but with more verbose comments and different code structure. We aim at achieving online depth reconstruction & recording, these real-time sensing data can be interpreted and learned as an immediate tactile feedback, with abundant knowledge on objects the sensor touched.

Test image source: [gelsight_heightmap_reconstruction](https://github.com/siyuandong16/gelsight_heightmap_reconstruction). We will collect own data from our sensor soon.

## How to use it
First, clone this repo and install required packages:
```
pip install -r requirements.txt
```
Then run `calib.py` , place the ball, and apply indent force to produce images for sensor calibration. You should adjust calibration parameters(the ball radius, pixel-to-mm rate) before this step. This calibration process will produce a `.npz` lookup table. This script would ask user for manually circle search, with (`I`, `J`, `K`, `L` for coarse movement, `W`, `A`, `S` and `D` for fine movement) to adjust the location of center and (`M` and `N`) to increase/decrease the radius of lookup circle. Once circle search is finished, the pixels within the circle range will be extracted in (R, G, B) pair and assigned to a gradient value evaluated from geometry parameters.


You can run `lookup.py` to display the constructed surface, provided the lookup table, the background image and the image you want to extract depth from are presented. You can run `depth_camera_demo.py` to see the depth reconstruction in real time, but please be noted the constructed surface will not be displayed because the construction from point-set is time consuming.

**update**: Now you can run `lookup_cuda.py` for faster reconstruction. This script will load the lookup table into GPU memory, and lookup operation will be interpreted as CUDA operation. To use this script, you need to install `open3D` lib with correct CUDA version. We tried this script on our homemade sensor, it can run as fast as ~80 FPS at 480x480 resulotion.
...
![DSC01474](https://github.com/user-attachments/assets/196ff60a-9c2c-4038-ad6b-4393ed45c9a6)


<img width="1297" height="869" alt="image" src="https://github.com/user-attachments/assets/2e79fdd1-e242-49a4-bffb-2cb8c5807fd0" />
