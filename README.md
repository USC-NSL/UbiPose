
# UbiPose: Towards Ubiquitous Outdoor AR Pose Tracking using Aerial Meshes

This is an implementation of the system described in the paper *UbiPose: Towards Ubiquitous Outdoor AR Pose Tracking using Aerial Meshes*

## To download

```
git clone --recurse-submodules https://github.com/USC-NSL/UbiPose.git
```

This will also download the submodules.


## Environment setup

We recommend using docker to setup the developement environment. Two Dockerfiles are provided. 

To build the docker image please run
### On a CUDA desktop
Please install docker and required NVIDIA container [toolkits](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker).
```
docker build -t ubipose .
```

To run the container and mount the ubipose repo to the container, run the following command. The privileged flag is needed to enable EGL headless rendering.
```
docker run --gpus all --privileged --rm -it -v ${PATH_TO_UBIPOSE}:/ubipose --name ubipose ubipose
```

### On a Jetson NX
To install the required environment to run NVIDIA containers on Jetson, please follow the prerequisites at this [link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack). 
```
docker build -t ubipose -f Dockerfile.jetson .
```

To run the container and mount ubipose repo to the container, run the following command. Notice the way to enable GPU is slightly different than the desktop version (tested on Jetson NX with JetPack 5.0.2)
```
docker run --runtime nvidia --privileged --rm -it -v ${PATH_TO_UBIPOSE}:/ubipose --name ubipose ubipose
```


## Compilation
Go to Ubipose's project folder

Run the following commands in your container

```
mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=Release .. && \
make -j
```

## Run

### Prepare the mesh

Please follow this tutorial [video](https://www.youtube.com/watch?v=My5HoPfOxfg) for extracting aerial meshes using [RenderDoc](https://renderdoc.org/builds), [Blender](https://www.blender.org/download/lts/) and [MapModelImporter](https://github.com/eliemichel/MapsModelsImporter). Please refer to MapModelImporter's [release](https://github.com/eliemichel/MapsModelsImporter/releases) page for a correct combination of software versions.

### Prepare the data

Please download the city.zip and decompress it under ```data/``` folder. The folder structure of ```data/``` should look like this.

```
├── data
│   ├── city
│   │   ├── arkit
│   │   ├── database.db
│   │   ├── reconstruction
│   │   │   ├── cameras.bin
│   │   │   ├── images.bin
│   │   │   ├── points3D.bin
│   │   │   └── project.ini
│   │   ├── san_jose_dt
│   │   ├── transformed
│   │   │   ├── cameras.txt
│   │   │   ├── images.txt
│   │   │   └── points3D.txt
│   │   └── transform.txt
```

Under the project folder (NOT the build folder):

### On a CUDA desktop
```
./build/ubipose/ubipose_pipeline_main_ios_data --arkit_directory ./data/city/arkit/ --config_file=./configs/ubipose_controller_city.yaml  --use_aranchor=false --start_timestamp=1678565810 --end_timestamp=1678566005 
```

### On a Jetson NX
```
./build/ubipose/ubipose_pipeline_main_ios_data --arkit_directory ./data/city/arkit/ --config_file=./configs/ubipose_controller_city_nx.yaml  --use_aranchor=false --start_timestamp=1678565810 --end_timestamp=1678566005 
```

### Note about running the pipeline
Running the code for the first time might take a significantly long time. This is because the NVIDIA's TensorRT framework optimizes the SuperPoint and SuperGlue models in to the TensorRT engines for your GPU. We try to provide the corresponding engine files, but it's up to TensorRT's heuristics to determine if our engine file could run on your GPU hardware. It can take up to **30min** (on a CUDA desktop) or **60min** (on a Jetson NX) to finish the conversion. After the first run, the program skips the the conversion process in future runs (unless the input sizes change).

## Evaluation

Install the required libraries for the evaluation script
```
cd python && python3 -m pip install -r requirements.txt
```

**Under the project folder**

Run the evaluation script:
```
python3 python/ubipose_stats.py --colmap_image_txt data/city/transformed/images.txt --results result.csv --stats stats.csv
```

Expected output:

```
Results for file result.csv:
Median errors: 0.632m, 0.870deg
Percentage of test images localized within:
        1cm, 1deg : 0.51%
        2cm, 2deg : 0.51%
        3cm, 3deg : 0.51%
        5cm, 5deg : 1.54%
        25cm, 2deg : 7.18%
        50cm, 5deg : 32.82%
        500cm, 10deg : 100.00%
95th translation error = 1.065973
95th rotation error = 1.243292
99th translation error = 1.269447
99th rotation error = 1.485843
Percentage of early exited frames 0.113402
Number of cache localized frames 95
Percentage of cache localized frames 0.489691
Median latency = 364.00ms, 95th latency = 852.70ms
```

Please ignore the warning of numeric overflow like the following. This is due to the uninitialized VIO result.
```
python/ubipose_stats.py:33: RuntimeWarning: overflow encountered in scalar power
  1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
python/ubipose_stats.py:39: RuntimeWarning: overflow encountered in scalar power
  1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
/usr/local/lib/python3.8/dist-packages/numpy/core/fromnumeric.py:1774: RuntimeWarning: invalid value encountered in reduce
  return asanyarray(a).trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)
Number of early exited frames 22
```
