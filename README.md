# goldfish-detector
🐟 The `FishDetector` class is a Python class used to detect fish in video frames. 
* It can process frames from a webcam or a video file as input, and uses color filtering, morphological transformations, and contour detection to detect the fish in the frames. 
* It counts the number of fish moving to the right and to the left, and displays the final result on the screen. 
* The class uses the OpenCV library to process the frames and detect the fish. 
* It uses a thread pool to read and process the frames in parallel, which can improve the speed at which the frames are processed on systems with multiple CPU cores.

# Output
![output](https://user-images.githubusercontent.com/84540554/209397137-c3750a85-fe0d-47bd-8fee-cab2b7041a23.gif)
* Better edge-detection
* Blazingly fast (detection only takes around 0.1 - 0.2s for ~250 frames)
* multithreaded

# Why?
### Because I saw [Michael Reeves' video on letting his goldfish trade stocks for him](https://www.youtube.com/watch?v=USKD3vPD6ZA) and I wanted to create my own version of detecting orange blobs
![image](https://user-images.githubusercontent.com/84540554/209398374-656aece0-7ac2-480a-b47f-7bfa4853054b.png)
![image](https://user-images.githubusercontent.com/84540554/209398545-26dd3a25-93b2-4206-885c-d11b4024249b.png)
![image](https://user-images.githubusercontent.com/84540554/209398571-29276575-079d-4e99-8d2a-4c983944d002.png)
![image](https://user-images.githubusercontent.com/84540554/209398505-301fbcd8-5e04-4f8f-b635-1260d74a022a.png)

# Features
* It has 2 modes:
* * Video: Upload video(s) on the same directory and it'll automatically detect and let you pick one of them
* * Camera mode: Use your webcam to detect "goldfishes" (terrible and wouldn't recommend!)
* Draws a rectangle over big orange blobs and displays the current position
* Shows which side is currently winning on each frame
* Shows the final result
* Grayscales the winning side

# Libraries used
### [OpenCV](https://opencv.org/)

# How the code works:
### The `FishDetector` class is a class that uses computer vision techniques to detect fish in video frames. It has several methods:

* __init__: This is the constructor method for the class. It initializes the instance variables that will be used throughout the class.

* find_files: This method searches a given directory for files with a specified suffix (".mp4" by default) and stores the paths to these files in a list.

* getInput: This method prompts the user to specify whether they want to use a webcam or a video file as the input for the fish detection process, and gets the necessary information from the user (e.g., the number of frames to take from the webcam or the path to the video file).

* read_frames: This method reads in the video frames from either the webcam or the specified video file. It uses a thread pool to parallelize the frame reading process.

* read_frame: This method reads in a single frame from either the webcam or the video file. It is called by the read_frames method to read in each frame.

* detect_fish: This method uses computer vision techniques to detect fish in the video frames. It processes the frames in parallel using a thread pool and counts the number of fish moving in each direction (left or right).

* process_frame: This method processes a single frame to detect fish in it. It is called by the detect_fish method to process each frame.

* show_frames: This method displays the processed frames to the user.

* run: This is the main method of the class that coordinates the various steps involved in the fish detection process (e.g., getting input from the user, reading in the frames, detecting the fish, and displaying the processed frames).

## How OpenCV aids in detecting the fish
The detect_in_frame method uses several functions from the OpenCV library to filter the frame, apply morphological transformations, and detect contours in the frame. Specifically, it uses the following functions:

* cv2.inRange: This function filters the frame to only keep the pixels within the specified color range (specified by the lower_bound and upper_bound instance variables).
* cv2.findContours: This function detects the contours in the filtered and transformed frame.
The detect_fish method uses the cv2.setNumThreads function to specify the number of threads to use for processing the frames in parallel.

Finally, the display_result method uses the cv2.imshow function to display the final result on the screen.

## Some of the drawbacks
* The class relies on the user to choose the appropriate color range for the fish. If the fish are not within the specified color range, they will not be detected.

* The class may not work well for videos with low lighting conditions, as the color filtering may not be effective in these cases.

* The class may not work well for videos with a lot of noise or clutter, as the morphological transformations may not be effective in removing all the noise.

* The class uses a fixed set of morphological transformations (erosion followed by dilation) to remove noise. This may not be the most effective approach for all types of noise.

* The class counts the number of fish moving to the right and to the left by detecting contours in the frames and checking their positions. This may not be the most accurate way to count the fish, as the contours may not always accurately reflect the shape and movement of the fish.

* The class uses a fixed frame size (320x240) and does not allow the user to choose a different frame size. This may not be suitable for all types of videos.

* The class uses a fixed number of threads (8) for processing the frames in parallel. This may not be the optimal number of threads for all types of videos and hardware configurations.

* The class does not handle errors and exceptions effectively, and may crash if an error occurs during the processing of the frames.

## Efficiency focused detection:
* The class uses a thread pool to read the frames from the input source in parallel, which can improve the speed at which the frames are read.

* The class uses the cv2.setNumThreads function to specify the number of threads to use for processing the frames in parallel. This can improve the speed at which the frames are processed, especially on systems with multiple CPU cores.

* The class uses morphological transformations to remove noise from the frames, which can be faster than more complex noise reduction techniques.

* The class uses contour detection to detect the fish in the frames, which can be faster than more complex object detection techniques.

* The class uses a fixed frame size (320x240) and a fixed number of frames (250 for webcam input, all frames for video file input), which can reduce the overall processing time compared to using larger frame sizes or more frames.

* The class only processes the frames once, and stores the result in the result instance variable. This allows the result to be displayed multiple times without the need to process the frames again.

### However, it is important to note that the overall efficiency of the FishDetector class will depend on several factors, including the input source, the hardware configuration, and the complexity of the processing tasks. In some cases, the class may not be the most efficient solution for detecting fish in video frames.

## Scalability
The scalability of a system refers to its ability to handle increasing amounts of work or data without a decrease in performance. The scalability of the FishDetector class will depend on several factors, including the input source, the hardware configuration, and the complexity of the processing tasks.

If the input source is a webcam, the class is designed to process a fixed number of frames (up to 250) specified by the user. The scalability of the class in this case will depend on the speed at which the frames can be read from the webcam and the speed at which they can be processed.

If the input source is a video file, the class is designed to process all the frames in the file. The scalability of the class in this case will depend on the size of the video file and the speed at which the frames can be read and processed.

In both cases, the class uses a thread pool to read and process the frames in parallel, which can improve the scalability of the class on systems with multiple CPU cores. However, the overall scalability of the class will also depend on the complexity of the processing tasks and the efficiency of the algorithms used.

It is important to note that the scalability of the FishDetector class as implemented may not be optimal for all types of videos and hardware configurations. For example, using larger frame sizes or more complex processing tasks may decrease the scalability of the class. Similarly, using a smaller number of threads may decrease the scalability of the class on systems with multiple CPU cores.

# Previous Iterations
## Figuring out opencv to draw rectangles on orange blobs
![image](https://user-images.githubusercontent.com/84540554/209396549-d7c9348e-b75a-4235-8cce-0adff72aae18.png)

## Drawing boxes on a frame of the fish
![image](https://user-images.githubusercontent.com/84540554/209396640-f8d54a3c-2349-4619-87f2-d09e3bb7a6b4.png)

## Using the class on a video for the first time
![output](https://user-images.githubusercontent.com/84540554/209397412-bc049c4e-dff6-46e4-bb0f-ae998b838d46.gif)

## Previous commit
![output](https://user-images.githubusercontent.com/84540554/209322451-8f8b1344-17ea-49ff-9c08-98f6b88dbbe1.gif)
