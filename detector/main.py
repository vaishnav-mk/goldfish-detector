import cv2
import numpy as np
import os
import pathlib


class FishDetector:
    def __init__(self):
        self.vid = None
        self.inputMethod = None
        self.numFrames = None
        self.videoPath = None
        self.frames = []
        self.new_frames = []
        self.lower_bound = np.array([200, 50, 0])
        self.upper_bound = np.array([254, 254, 254])
        self.files = []

    def find_files(self, path, suffix=".mp4"):
        if os.path.isfile(path) and path.endswith(suffix):
            self.files.append(path)
        elif os.path.isdir(path):
            for subpath in os.listdir(path):
                self.find_files(os.path.join(path, subpath), suffix)

    def getInput(self):
        while True:
            inputMethod = input("Enter 'cam' to use webcam or 'vid' to use video: ")
            if inputMethod.startswith("c") or inputMethod.startswith("v"):
                self.inputMethod = 0 if inputMethod.startswith("c") else 1
                break
            else:
                print("Invalid input")

        if inputMethod.startswith("c"):
            while True:
                numFrames = input(
                    "Enter the number of frames to take (max 250, min 5): "
                )
                if numFrames.isdigit() and 5 <= int(numFrames) <= 250:
                    self.numFrames = int(numFrames)
                    break
                else:
                    print("Invalid input")

        elif inputMethod.startswith("v"):
            self.find_files(pathlib.Path(__file__).parent.absolute())
            videos = {str(i + 1): file for i, file in enumerate(self.files)}
            if len(videos) == 0:
                print(
                    "No videos found in the current directory: Upload a video and try again or use webcam mode instead!"
                )
                exit()
            while True:
                print("Found the following videos:")
                for i, file in enumerate(self.files):
                    print(f"{i + 1}: {os.path.basename(file)}")
                video = input("Enter the number of the video you want to use: ")
                if video.isdigit() and 1 <= int(video) <= len(videos):
                    self.videoPath = videos.get(video)
                    break
                else:
                    print("Invalid input")

    def read_frames(self):
        if self.inputMethod == 0:
            print("videoPath: ", self.videoPath)
            for i in range(self.numFrames):
                success, frame = self.vid.read()
                if success:
                    self.frames.append(frame)
                else:
                    print("Video not found")
                    exit()
        elif self.inputMethod == 1:
            while True:
                success, frame = self.vid.read()
                if success:
                    self.frames.append(frame)
                else:
                    break

    def detect_fish(self):
        result = {}
        for frame in self.frames:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = cv2.inRange(rgb_img, self.lower_bound, self.upper_bound)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
            for index, contour in enumerate(contours):
                cv2.line(
                    frame,
                    (frame.shape[1] // 2, 0),
                    (frame.shape[1] // 2, frame.shape[0]),
                    (0, 255, 0),
                    2,
                )

                if cv2.contourArea(contour) >= frame.size:
                    print("Contour area is greater than image size")
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Located at: ({}, {})".format(x, y),
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                if x + w // 2 > frame.shape[1] // 2:
                    frame = self.grayscale_part(frame, "right")
                    result["right"] = result.get("right", 0) + 1
                    self.new_frames.append(frame)

                else:
                    frame = self.grayscale_part(frame, "left")
                    result["left"] = result.get("left", 0) + 1
                    self.new_frames.append(frame)
        return result

    def grayscale_part(self, frame, side):
        left, right = frame[:, : frame.shape[1] // 2], frame[:, frame.shape[1] // 2 :]
        if side == "left":
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                left,
                "FISH IS ON THE LEFT",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        elif side == "right":
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                right,
                "FISH IS ON THE RIGHT",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        final = np.concatenate((left, right), axis=1)
        return final

    def create_video(self, result):
        blank_frame = np.zeros(
            (self.new_frames[0].shape[0], self.new_frames[0].shape[1], 3), np.uint8
        )
        cv2.putText(
            blank_frame,
            "WINNER: {}".format(max(result, key=result.get)),
            (blank_frame.shape[1] // 2 - 100, blank_frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0)
            if result.get("left", 0) > result.get("right", 0)
            else (0, 0, 255),
            3,
        )
        cv2.putText(
            blank_frame,
            "LEFT: {} - RIGHT: {}".format(
                result.get("left", 0), result.get("right", 0)
            ),
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
            if result.get("left", 0) > result.get("right", 0)
            else (0, 0, 255),
            2,
        )
        for i in range(30):
            self.new_frames.append(blank_frame)
        height, width, layers = self.new_frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, size)
        print("Displaying frames... (check your taskbar for the video)")
        for frame in self.new_frames:
            window_name = "Output.mp4"
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            out.write(frame)
        out.release()

    def run(self):
        print("🐟 | Welcome to the Fish Detector! (currently detects blobs of orange)")
        self.getInput()
        if self.inputMethod == 0:
            self.vid = cv2.VideoCapture(0)
        elif self.inputMethod == 1:
            self.vid = cv2.VideoCapture(self.videoPath)
        print("Reading frames...")
        self.read_frames()
        print(f"Read {len(self.frames)} frames\nDetecting fish...")
        result = self.detect_fish()
        print(f"Detected {len(self.new_frames)} frames\nCreating video...")
        self.create_video(result)
        print("Video created!")
        print(f"Done! Result saved to {os.getcwd()}\output.mp4")


if __name__ == "__main__":
    fd = FishDetector()
    fd.run()
