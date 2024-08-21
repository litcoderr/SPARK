# SPARK: multi-vision Sensor Perception And Reasoning benchmarK 
A benchmark dataset and simple code examples for **measuring the perception and reasoning of multi-sensor Vision Language models**.

# Problems

<p align="center">
  <img src="resources/problems.png" :height="300px" width="600px">
</p>
Many Vision-Language models have basic knowledge of various sensors such as thermal, depth, and X-ray, **but they do not attempt to view images while understanding the physical characteristics of each sensor.** Therefore, we have created a benchmark that can measure the differences between images and multi-vision sensor information.

# Dataset
<p align="center">
  <img src="resources/examples.png" :height="400px" width="800px">
</p>
The benchmark dataset consists of **four types of sensors(RGB, Thermal, Depth, X-ray) and six types of questions(Existence, Count, Position, Scene Description, Contextual Reasoning, Sensor Reasoning)**. Examples of each are shown in the image above.

download link : https://huggingface.co/datasets/topyun/SPARK
