# GelSight-homemade
A homemade version of MIT gelsight, with classic gradient reconstruction.

Gelsight is a visual tactile sensor used for surface reconstruction, which was introduced by MIT in their [CVPR paper](https://doi.org/10.1109/CVPR.2009.5206534). Our code is similar to [this repo](https://github.com/siyuandong16/gelsight_heightmap_reconstruction), but with more verbose comments and different code structure. We aim at achieving online depth reconstruction & recording, these real-time sensing data can be interpreted and learned as an immediate tactile feedback, with abundant knowledge on objects the sensor touched.

Test image source: [gelsight_heightmap_reconstruction](https://github.com/siyuandong16/gelsight_heightmap_reconstruction). We will collect own data from our sensor soon.

## How to use it
First, clone this repo and install required packages:
```
pip install -r requirements.txt
```
Then run `calib.py` , place the ball, and apply indent force to produce images for sensor calibration. This calibration process will produce a `.npz` lookup table, with specified geometry parameters of the sensor(i.e. `mm2Pixel`) and the ball(i.e. its radius). This script would ask user for manually circle search, with (`W`, `A`, `S` and `D`) to adjust the location of center and (`M` and `N`) to increase/decrease the radius of lookup circle. Once circle search is finished, the pixels within the circle range will be extracted in (R, G, B) pair and assigned to a gradient value evaluated from geometry parameters.

...

## Workflow:
<img width="1070" height="446" alt="image" src="https://github.com/user-attachments/assets/2fd19d90-f24a-40c5-9bf8-5cbe2a3747c3" />

<img width="1070" height="555" alt="image" src="https://github.com/user-attachments/assets/b84c3c86-eee2-4dee-b09b-5636b9c86d60" />

<img width="1070" height="1038" alt="image" src="https://github.com/user-attachments/assets/156a8b28-d174-4b29-83bf-3af3dd22f10b" />
