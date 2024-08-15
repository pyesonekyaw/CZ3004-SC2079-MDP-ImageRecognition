<br />
<p align="center">
  <img src="/images/Eye.png" alt="Logo" height=150 >
  <h1 align="center">
    CZ3004/SC2079 Multidisciplinary Project - Image Recognition
  </h1>
</p>

## About
2024 August Update: Someone sent me the slides from the briefing of this semester, this repository, along with my other MDP-related ones, are entirely STILL reusable as far as I can see. SCSE can become CCDS but MDP is still MDP. As usual, retrain the YOLO model (or use something more recent la). Once again, that is a 1-day thing. If you are using these repositories and you don't have a functioning, fully-integrated system by end of Week 4, reconsider your life choices and your peer evaluations.

2023 Semester 1 Update: At least from what my juniors told me, this repository, along with my other MDP-related ones, are entirely reusuable. The only exception is that you will need to retrain the YOLO model since the fonts/colors were changed. That is a 1-day thing. If you are using these repositories and you don't have a functioning, fully-integrated system by end of Week 4, reconsider your life choices.

Y'all, if you are using this code, which apparently a LOT of y'all are, at least star this repo leh

This repository contains the code for the image recognition component of the CZ3004/SC2079 Multidisciplinary Project. The repository is responsible for the following:
- Data collection using Raspberry Pi camera
- Training pipeline for YOLOv5, v7 and v8 models
- Inference server for performing inference on the images captured by the robot to identify the symbols and stitching the images together to form a summary of the results

To get semantics out of the way, yes, this is actually an object detection task. But for some reason, NTU has called it image recogntion, so I will be using the term `image recognition` throughout this repository. 

## Overall Pipeline
<img src="/images/Pipeline.png" alt="Pipeline" width="850" >

The overall pipeline for image recognition consists of the following steps:
1. Supervised Pretraining: Train models using past semester's data which is freely available online through the Roboflow Universe platform.
2. Data Collection: Collect data using the Raspberry Pi camera for edge cases or changes in color of the symbols.
3. Semi-Supervised Data Annotation: Annotate the data collected using the YOLOv5 model trained in Step 1.
4. Model Training/Finetuning: Finetune the models using the annotated data.
5. Heuristics: Use heuristics to improve the performance of the models for the specific tasks
6. Camera Calibration: Calibrate the camera to improve ensure visibility of the symbols even under adverse conditions like harsh sunlight or dark rooms

I will go over each step in detail below.

### 1. Supervised Pretraining
At first, I was intending to do some actual unsupervised pretraining but then I came upon the wealth of already painstakingly annotated data available online. I decided to use this data to pretrain the models instead, in a fully supervised manner. What I mean by this is that I just trained the models normally using the training notebooks, nothing special was done. So, I guess you could say the term 'pretraining' is used rather loosely. But hey, buzzwords make the video submission stand out, right?

The data is available on the [Roboflow Universe](https://universe.roboflow.com/) platform. I have included the list of datasets I downloaded in the Documents folder of this repository. Not all the datasets were eventually used for pretraining. Altogether, I used around 100,000 images, with each image having a single symbol most of the time, for pretraining.

I also came upon some pretrained models from previous semester's groups, which I also used as a pretrained model for the pretraining. 

### 2. Data Collection
I used the Raspberry Pi camera to collect data. The data collection was done both indoors and outdoors, from a variety of angles, distances and lighting conditions. I also collected data for the edge cases, such as when the symbols are not properly illuminated or when the symbols are partially obstructed. The script I used is provided in the `Data Collection Scripts` folder. But you could also just take a video with the Raspberry Pi camera, and then extract the frames from the video.

Luckily, for my semester, the color of the symbols and the symbols themselves were not changed, so I did not have to collect too much data, just the edge/hard cases. From what I've seen on RoboFlow, there are two main sets of colors, and tons of data for both sets are available on RoboFlow, so you can probably just find the correct colors there and do pretraining. 

I also found that even if the colors were changed, the models were still able to perform well (of course no longer near-perfect, but still very good), so I suspect you only need to collect a small amount of data to do finetuning on.

In my case, I collected around 10,000 images of edge/hard cases. 

### 3. Semi-Supervised Data Annotation
Now that we have the data, we need to annotate it next. I saw videos of groups of yester-semesters doing really painstaking manual annotation, and I was like, "No way I'm doing that". So I decided to use the YOLOv5 model trained in Step 1 to do the annotation for me.

<img src="/images/Annotation.png" alt="Annotation" width="850" >

Once again, like how I used the term 'pretraining' loosely, I am using the term 'semi-supervised' rather loosely as well. But this method is something that has been used by both research and industry to do data labelling for a long time. The annotation pipeline is as follows:
1. Use the YOLOv5 model trained in Step 1 to predict the bounding boxes for the images in the dataset.
2. Check through the predictions and correct any errors made by the model.

As the symbol recognition/detection task is incredibly easy, I did not have to do much correction, everything was near-perfect.

### 4. Model Training/Finetuning

Then, I just combined the pretraining and the annotated data, and trained the models using the training notebooks.

Only one word of caution here, be sure to turn off the horizontal flip image augmentation which is enabled by default, as we do not want the model to be confused for the left and right arrow symbols. 

Training was done on Google Colab free tier. Ultralytics' library has options to save and resume training, so I just trained the models for a few hours each day, and resumed training the next day, so that I don't hit the time limit for the free tier. You DO NOT need the paid tier for this. This module is not worth spending money on. At worst, just borrow a few Google accounts to train multiple models at the same time.

I trained some of the models for 100 epochs, but I find that training around 20 epochs is more than enough for near-perfect performance. 

### 5. Heuristics 
With the models trained, I just plugged them into the inference server pipeline (provided in the folder) and tested them out with the actual robot and camera. But of course, the reality is that there can be multiple symbols that are in the image captured by the robot, so you have to do some filtering for that. The heuristics/filters I used are as follows, from the simplest to the most unnecessary:

1. Ignore the bullseyes 
2. Filter by bounding box size - take the symbol with the largest bounding box. Be careful for certain symbols like `1` which will have a tiny bounding box, so be sure to account for that like I have.
<img src="/images/Heuristic2.png" alt="Heuristic2" width="850" >
3. Filter by signal from algorithm 
    * If algorithm says the robot should be directly in front of the symbol - Algorithm passes along a `C` signal in the command for capturing the image
    * The same goes for `L` and `R` for left and right
    * This is only when the other filters like the size do not work to filter out the other symbols, i.e bounding boxes are of similar sizes

<img src="/images/Heuristic1.png" alt="Heuristic1" width="850" >

### 6. Camera Calibration
Instead of using PiCamera which does not allow for finetuned calibration, I used LibCamera which allows for more control over the camera. I used the GUI from the following repository to calibrate the camera: [Pi_LibCamera_GUI](https://github.com/Gordon999/Pi_LIbCamera_GUI)

Please follow the instructions there to calibrate the camera. I created different calibration config files for different scenarios such as indoors, outdoors, and harsh sunlight. As calibration will be different for each camera hardware, I did not include the config files in this repository.

Since LibCamera is used to calibrate the camera, it is also used to capture the images with the given configuration file. I find that the exposure and metering mode are the most important parameters to tune for the camera. 

## Model 

### Model Architecture
I experimented with YOLOv5, v7, and v8, and found all three to give roughly the same performance and same latency for inference. So, I just used v5 in the end. The model architecture really does not matter in this task as long as it is modern enough. 

### Model Weights

Weights for YOLOv5, v7, and v8 models are available in the `Weights` folder. The weights are provided as-is, and I make no guarantees about their performance. However, from my memory, the weights should be able to achieve >95% F1 score on any test set you use that has symbols properly illuminated and visible, assuming the symbol colors do not change. I am not providing any specific comparisons or metrics as the performance is dependent on the test set used, and let's face it, the task is a very simple one, so most of the time, the models will be able to perform well.

`Week_8.pt` and `Week_9.pt` are the models that I eventually used for the actual tasks. Both are just YOLOv5 models. Week 8 models are trained on all symbols, while Week 9 models are only trained on left, right and bullseye symbols. `Week_9.pt` is further finetuned on a harsh sunlight dataset that I collected, so it should be able to perform much better outdoors.

## Datasets
I have not included the datasets I used/collected here, but if you do need them, feel free to contact me.

## Inference Server (only for YOLOv5)

```bash
pip install -r requirements.txt
```

Start the server by

```bash
python main.py
```

The server will be running at `localhost:5000`

You can modify the code easily to adapt it for v7 or v8 models if you need to.

Some miscellaneous notes:
- Raw images from Raspberry Pi are stored in the `uploads` folder.
- After calling the `image/` endpoint, the annotated image (with bounding box and label) is stored in the `runs` and `own_results` folder.
- After calling the `stitch/` endpoint, two stitched images using two different functions (for redundancy) are saved at `runs/stitched.jpg` and in the `own_results` folder.

### API Endpoints:

##### 1. POST Request to /image

The image is sent to the API as a file, thus no `base64` encoding required.

**Sample Request in Python**

```python3
response = requests.post(url, files={"file": (filename, image_data)})
```

- `image_data`: a `bytes` object

The API will then perform three operations:

1. Save the received file into the `/uploads` and `/own_results` folders.
2. Use the model to identify the image, save the results into the folders above.
3. Return the class name as a `json` response.

**Sample JSON response**

```json
{
  "image_id": "D",
  "obstacle_id": 1
}
```

Please note that the inference pipeline is different for Task 1 and Task 2, be sure to comment/uncomment the appropriate lines in `app.py` before running the API.

##### 2. POST Request to /stitch

This will trigger the `stitch_image` and `stitch_image_own` functions.

- Images found in the `run/` and `own_results` directory will be stitched together and saved separately, producing two stitched images. We have two functions for redundancy purposes. In case one fails, the other can still run.

# Disclaimer

I am not responsible for any errors, mishaps, or damages that may occur from using this code. Use at your own risk. The code and models are provided as-is, with no warranty of any kind. 

# Acknowledgements

I used Group 28's inference server as a boilerplate and augmented the inference pipeline heavily with the heuristics. 
- [Group 28](https://github.com/CZ3004-Group-28)

Naturally, I also used Ultralytics' libraries and training notebooks for the models.
- [Ultralytics](https://github.com/ultralytics)

# Related Repositories

* [Website](https://github.com/pyesonekyaw/MDP-Showcase)
* [Algorithm](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-Algorithm)
* [Simulator](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-Simulator)
* [Raspberry Pi](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-RaspberryPi)
* [HuggingFace Space](https://huggingface.co/spaces/pyesonekyaw/Image_Recognition-CZ3004_SC2079_Multidisciplinary_Project-NTU_SG)
* [Task 1 Model](https://huggingface.co/pyesonekyaw/MDP_ImageRecognition_YOLOv5_Week_8_AY22-23_NTU-SG)
* [Task 2 Model](https://huggingface.co/pyesonekyaw/MDP_ImageRecognition_YOLOv5_Week_9_AY22-23_NTU-SG)
