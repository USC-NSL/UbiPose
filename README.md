
# UbiPose: Towards Ubiquitous Outdoor AR Pose Tracking using Aerial Meshes

This is an implementation of the paper *UbiPose: Towards Ubiquitous Outdoor AR Pose Tracking using Aerial Meshes*

## To download

This will also download the submodules.

```
git clone --recurse-submodules https://github.com/USC-NSL/UbiPose.git
```

## Environment setup

We recommend using docker to setup the developement environment. Two Dockerfiles are provided. 

To build the docker image please run
### On a CUDA desktop
```
docker build -t ubipose .
```

To run the container and mount ubipose repo to the container. The privileged flag is needed to enable EGL headless rendering.
```
docker run --gpus all --privileged --rm -it -v ${PATH_TO_UBIPOSE}:/ubipose --name ubipose ubipose
```

### On a Jetson NX

```
docker build -t ubipose -f Dockerfile.jetson .
```

To run the container and mount ubipose repo to the container. Notice the way to enable GPU is slightly different than the desktop version (tested on Jetson NX with JetPack 5.0.2)
```
docker run --runtime nvidia --privileged --rm -it -v ${PATH_TO_UBIPOSE}:/workspace --name ubipose ubipose
```


## To build
Go to Ubipose's project folder

Run the following commands in your container

```
mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release .. && \
make -j
```

## To run
Under the project folder (NOT the build folder)

### On a CUDA desktop
```
./build/ubipose/ubipose_pipeline_main_ios_data --arkit_directory ./data/city/arkit/ --config_file=./configs/ubipose_controller_city.yaml  --use_aranchor=false --start_timestamp=1678565810 --end_timestamp=1678566005 
```

### On a Jetson NX
```
./build/ubipose/ubipose_pipeline_main_ios_data --arkit_directory ./data/city/arkit/ --config_file=./configs/ubipose_controller_city_nx.yaml  --use_aranchor=false --start_timestamp=1678565810 --end_timestamp=1678566005 
```

### Note about running the pipeline
Running the code for the first time might take significant long time. This is because the NVIDIA's TensorRT framework is optimizing the SuperPoint and SuperGlue models in to the TensorRT engines for your GPU. We try to provide the corresponding engine files, but it's up to TensorRT's heuristic to determine if our engine file could run on your GPU hardware. It would take up to **30min** (on a CUDA desktop) or **60min** (on a Jetson NX) to finish the conversion. After the first run, the program skip the the conversion process in the future run (unless the input sizes changed).

## To evaluate accuracy and runtime

Install the required libraries for the evaluation script
```
cd python && python3 -m pip install -r requirements.txt
```

**Under the project folder**

Run the evaluation script:
```
python3 python/meshloc_stats.py --colmap_image_txt data/city/transformed/images.txt --results result.csv --stats stats.csv
```


