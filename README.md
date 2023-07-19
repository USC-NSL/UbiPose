
## Environment setup

We recommend using docker to setup the developement environment. A dockerfile is provided. To build the docker image please run
```
docker build -t ubipose .
```

To run the container and mount ubipose repo to the container. The privileged flag is needed to enable EGL headless rendering.
```
docker run --gpus all --privileged --rm -it -v ${PATH_TO_UBIPOSE}:/ubipose --name ubipose ubipose
```

## To build
Run the following commands in your container

```
UbiPose# mkdir build && cd build
UbiPose/build# cmake -DCMAKE_BUILD_TYPE=Release ..
UbiPose/build# make -j
```

## To Run
Under the project folder (NOT the build folder)

```
UbiPose# ./build/ubipose/ubipose_pipeline_main_ios_data --arkit_directory ./data/city/arkit/ --config_file=./configs/ubipose_controller_city.yaml  --use_aranchor=false --start_timestamp=1678565810 --end_timestamp=1678566005 
```

## To evaluate error

Install the required libraries for the evaluation script
```
UbiPose# cd python
UbiPose/python# python3 -m pip install -r requirements.txt
```

To the evaluation script:
```
UbiPose# python3 python/meshloc_stats.py --colmap_image_txt data/city/transformed/images.txt --results result.csv --stats stats.csv
```



