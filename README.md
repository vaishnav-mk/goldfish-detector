# goldfish-detector
🐟 A goldfish detector with OpenCV


# Output
![output](https://user-images.githubusercontent.com/84540554/209322451-8f8b1344-17ea-49ff-9c08-98f6b88dbbe1.gif)

# Why?
### Because I saw [Michael Reeves' video on letting his goldfish trade stocks for him](https://www.youtube.com/watch?v=USKD3vPD6ZA) and I wanted to create my own version of detecting orange blobs

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
