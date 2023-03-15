<h1 align="center">
  Simultaneous Action Recognition and Human Whole-Body Motion and Dynamics Prediction from Wearable Sensors

</h1>


<div align="center">


K. Darvish, S. Ivaldi and D. Pucci, "Simultaneous Action Recognition and Human Whole-Body Motion and Dynamics Prediction from Wearable Sensors," 2022 IEEE-RAS International Conference on Humanoid Robots (Humanoids), Ginowan, Japan, 2022.
</div>

<p align="center">
  
https://user-images.githubusercontent.com/17707730/224910766-7cc169ce-e878-4b04-af71-e90023cc0431.mp4
  
</p>

<div align="center">
  <a href="#data"><b>Data</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#running"><b>Running</b></a> |
  <a href="https://ieeexplore.ieee.org/document/10000122"><b>Paper</b></a> |
  <a href="https://arxiv.org/abs/2303.07655"><b>arXiv</b></a> |
  <a href="https://youtu.be/uNs_L2X30xY"><b>Video</b></a>
</div>


# Data
To run the scripts, please download the required datasets and models provided in [https://zenodo.org/record/7731386#.ZBABX9LMJhE](https://zenodo.org/record/7731386#.ZBABX9LMJhE). After downloading the data, extract the zip file in the root directory of this repo. The `data` folder will have the following structure:

```
├── data
│   ├── README.md
│   ├── annotated-data
│   │   ├── ...
│   ├── models
│   │   ├── ...
│   ├── raw-data
│   │   ├── ...
│   ├── wearable-data
│   │   ├── ...
```

# Installation

## Requirements
-  Ubuntu 20.04.5 LTS (Focal Fossa)

## Installation: mamba \& robotology-superbuild 
- Install mamaba if you do not have:
  - follow the instructions provided in [here](https://github.com/robotology/robotology-superbuild/blob/master/doc/install-mambaforge.md)

- run the following command to create the enviornment for this code:
    ```sh
    cd <to the repo>
    mamba env create -f environment.yml
    mamba activate motion-prediction
    ```

- If you do not have robotology-superbuild installed in your system, follow the instructions in [here](https://github.com/robotology/robotology-superbuild/blob/master/doc/conda-forge.md) to install it in `motion-prediction` mamba env.
    - Activate the following profiles in robotology-superbuild if not already:
      - `ROBOTOLOGY_ENABLE_CORE`
      - `ROBOTOLOGY_ENABLE_DYNAMICS`
      - `ROBOTOLOGY_ENABLE_HUMAN_DYNAMICS`
      - `ROBOTOLOGY_USES_PYTHON`
  - More information about the installation of robotology-superbuild can be found in its [GitHub repo](https://github.com/robotology/robotology-superbuild).
  - remeber to source the robtology-superbuild by `source <robotology-superbuild path>/build/install/share/robotology-superbuild/setup.sh`.

-  in an activated and sourced environment try to run the following commands to ensure your environment is correctly setup:
    ```sh
    python
    >> import tensorflow as tf
    >> tf.__version__
    >> import yarp
    ```
**N.B.** <ins>Thereafter, all the terminals should be activated, sourced, and in the root folder of this repo.</ins>
## Installation of the project 

- build and test the python project:
  ```sh
  cd <motion-prediction path>
  pip install -e .
  pytest
  ```
- build and install the c++ modules by:
    ```sh
    cd <motion-prediction path>
    mkdir build
    ccmake ../
    # update the CMAKE_INSTALL_PREFIX to your desired directory
    make install
    ```

# Running
There are three phases in running the project: annotation, training, and testing.

## Annotation

### collect all the required data column-wise with desired frequency
- run yarp server
  ```sh
  yarpserver --write
  ```

- run collected wearable data using yarpdataplayer
  ```sh
  yarpdataplayer --withExtraTimeCol 2
  ```

- run IK solver to stream human states
  ```sh
  yarprobotinterface --config TransformServer.xml
  yarprobotinterface --config HumanStateProvider.xml
  ```
- run human motion data acquisition to collect human state and human dynamics data (feet force/torque interaction data) with the desired frequency (25 Hz, period 0.04 sec)
  ```sh
  humanDataPreparationModule --from humanDataAcquisitionForLogging.ini
  ```
- At the end of this stage, you should have a file containing time, human states, interction forces/torques, similar to data file in `data/raw-data/Dataset_2021_08_19_11_31_13.txt` .
### annotate the data
- run human motion data acquisition to annotate data and stream the vectorized human states (correctly set the path to the file saved in the previous step with variable `filePathToRead` in `src/humanMotionDataAcquisition/app/humanDataAcquisitionForAnnotation.ini`; remember to build and install the project)
  ```sh
  humanDataPreparationModule --from humanDataAcquisitionForAnnotation.ini
  ```
- in a new terminal, run human motion prediction visualizer:
  ```sh
  humanPredictionVisualizerModule --from HumanVisualizer.ini
  ```
- At the end of this stage, you should have a file containing columns with time, human states, interction forces/torques, and annotations, similar to data file in `data/annotated-data/Dataset_2021_08_19_20_06_39.txt`.

## Training
- to train a GMoE model, run the following script:
  ```sh
  python scripts/train.py
  ```
- remember to correctly set `data_path` variable to the path of the annotated data.
- you can modify the model and its parameters before training.
- save and close the plots during training as they are blocking the process.
- at the end of this script, you will see the results of LSTM and GMoE and the path to their saved models.

## Testing and Animation for Realtime Applications
### Real time scenario using wearables
- stream [human wearable data](https://github.com/robotology/wearables), either online or from collected data
  ```sh
  yarpdataplayer --withExtraTimeCol 2
  ```
- run IK solver
  ```sh
  yarprobotinterface --config TransformServer.xml
  yarprobotinterface --config HumanStateProvider.xml
  ```
- run the following command for vectorizing outputs
  ```sh
  humanDataPreparationModule --from humanDataStreamingTestOnline.ini
  ```
### Online scenario using logged data
- run the following command for vectorizing outputs
  ```sh
  humanDataPreparationModule --from humanDataStreamingTestFromLoggedFile.ini
  ```

### running test code for realtime prediction
- run the model for testing
  ```sh
  python scripts/test_moe.py
  ```
### visuaization of outputs
- you can run the following shell script to visualize the outputs:
  ```sh
  sh scripts/run_animations.sh
  ```



# Citing this work

If you find the work useful, please consider citing:

```bibtex
@INPROCEEDINGS{Darvish2022Simultaneous,
  author={Darvish, Kourosh and Ivaldi, Serena and Pucci, Daniele},
  booktitle={2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)}, 
  title={Simultaneous Action Recognition and Human Whole-Body Motion and Dynamics Prediction from Wearable Sensors}, 
  year={2022},
  volume={},
  number={},
  pages={488-495},
  doi={10.1109/Humanoids53995.2022.10000122}}

```

# Maintainer

This repository is maintained by:

| | | |
|:---:|:---:|:---:|
| [<img src="https://kouroshd.github.io/assets/img/kourosh_darvish_iit.jpg" width="40">](https://github.com/kouroshD) | [@kouroshD](https://github.com/kouroshD) | [pesonal web page](https://kouroshd.github.io/) |
