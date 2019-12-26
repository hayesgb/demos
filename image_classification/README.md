# Image Classification Using Distributed Training


This example demonstrates an end-to-end image classification solution using **[TensorFlow](https://tensorflow.org)**, **[Horovod](https://github.com/horovod/horovod#horovod)**, and **[Nuclio](https://github.com/nuclio/nuclio#nuclio---serverless-for-real-time-events-and-data-processing)**. The pipeline consists of 4 MLRun and Nuclio functions:

1. import an image archive from S3 to the cluster file system
2. tag images based on their name structure 
3. distrubuted training using TF, Keras and Horovod
4. automated deployment of a Nuclio model serving function (see **[Nuclio serving TF images](nuclio-serving-tf-images.ipynb)** and from a **[Dockerfile](./inference-docker)**)

<br><p align="center"><img src="workflow.png" width="600"/></p><br>

Also demonstrated is an **[automated pipeline](mlrun_mpijob_pipe.ipynb)** using MLRun and KubeFlow pipelines. 


## Notebooks & Code

* **[All-in-one: Import, tag, launch training, deploy serving](mlrun_mpijob_classify.ipynb)**
* **[Training function code](horovod-training.py)**
* **[Serving function development and testing](nuclio-serving-tf-images.ipynb)**
* **[Auto generation of KubeFlow pipelines workflow](mlrun_mpijob_pipe.ipynb)**
* **[Building serving function using Dockerfile](./inference-docker)**
  * **[function code](./inference-docker/main.py)**
  * **[Dockerfile](./inference-docker/Dockerfile)**

