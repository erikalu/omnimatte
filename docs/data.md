## Data
The data directory for a video is structured as follows:
```
video_name/
|-- rgb/
|   |-- 0001.png, ...
|-- mask/
|-- |-- 01, ...
|-- |-- |-- 0001.png, ...   
|-- flow/
|   |-- 0001.flo, ...
|-- confidence/
|   |-- 0001.png, ...
|-- homographies.txt
```
The `mask/` directory should contain a subdirectory for each omnimatte's input masks.

### Camera registration
The method requires as input homographies computed between frames (e.g. using OpenCV).

See `datasets/tennis/homographies.txt` for an example.

The expected format for `homographies.txt` is:
```
size: width height  # dimensions of video
bounds: x_min x_max y_min y_max   # world bounds
1 0 0 0 1 0 0 0 1   # homography for frame 1
... # homography for frame 2, etc.
```
After computing the homographies and saving to a text file,
the helper script `datasets/homography.py` can be used to compute the world bounds
and add the first 2 lines expected in the `homographies.txt` file:
```
python datasets/homography.py --homography_path path_to_homographies.txt --width vid_width --height vid_height
```
This will output `path_to_homographies-final.txt`, which should be renamed to `video_name/homographies.txt`.