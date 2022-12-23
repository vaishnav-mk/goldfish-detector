import cv2
import numpy as np
import os
import pathlib
import time
from multiprocessing.pool import ThreadPool


class FishDetector:
    def __init__(self):
        self.vid = None
        self.input_method = None
        self.num_frames = None
        self.video_path = None
        self.frames = []
        self.new_frames = []
        self.lower_bound = np.array([200, 50, 0])
        self.upper_bound = np.array([254, 254, 254])
        self.files = []
        self.result = {
            "right": 0,
            "left": 0,
        }

    def find_files(self, path, suffix=".mp4"):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(suffix):
                    self.files.append(entry.path)
                elif entry.is_dir():
                    self.find_files(entry.path, suffix)

    def getInput(self):
        while True:
            input_method = input("Enter 'cam' to use webcam or 'vid' to use video: ")
            if input_method.startswith("c") or input_method.startswith("v"):
                self.input_method = 0 if input_method.startswith("c") else 1
                break
            else:
                print("Invalid input")

        if input_method.startswith("c"):
            while True:
                num_frames = input(
                    "Enter the number of frames to take (max 250, min 5): "
                )
                if num_frames.isdigit() and 5 <= int(num_frames) <= 250:
                    self.num_frames = int(num_frames)
                    break
                else:
                    print("Invalid input")

        elif input_method.startswith("v"):
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
                    self.video_path = videos.get(video)
                    break
                else:
                    print("Invalid input")

    def read_frames(self):
        if self.input_method == 0:
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            with ThreadPool(processes=10) as pool:
                self.frames = pool.map(self.read_frame, range(self.num_frames))

        elif self.input_method == 1:
            vid = cv2.VideoCapture(self.video_path)

            vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            with ThreadPool(processes=8) as pool:
                while True:
                    success, frame = self.read_frame(vid)
                    if success:
                        self.frames.append(frame)
                    else:
                        break

    def read_frame(self, vid_or_index):
        if self.input_method == 0:
            success, frame = self.vid.read()
        else:
            success, frame = vid_or_index.read()
        return success, frame



    def detect_fish(self, num_threads=8):
        cv2.setNumThreads(num_threads)
        start = time.time()
        with ThreadPool(processes=num_threads) as pool:
            processed_frames = pool.map(self.detect_fish_in_frame, self.frames)
            for frame in processed_frames:
                self.new_frames.append(frame)
        print(f"Time taken: {round(time.time() - start, 2)}s")

    def detect_fish_in_frame(self, frame, scaling_factor=2, use_cropped_frame=False):
        frame = cv2.resize(frame, (0, 0), fx=1 / scaling_factor, fy=1 / scaling_factor)

        x, y, w, h = (0, 0, frame.shape[1], frame.shape[0] // 2)
        cropped_frame = frame[y : y + h, x : x + w]

        if use_cropped_frame:
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(rgb_frame, self.lower_bound, self.upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[
            :1
        ]  # get biggest contour

        for contour in contours:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
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
            cv2.drawContours(
                frame,
                contours,
                -1,
                (0, 0, 255),
                2,
            )
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
                self.result["right"] += 1
                frame = self.grayscale_part(frame, "right")
            else:
                self.result["left"] += 1
                frame = self.grayscale_part(frame, "left")
        return frame

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

    def create_video(self):
        result = self.result
        blank_frame = np.zeros(
            (self.new_frames[0].shape[0], self.new_frames[0].shape[1], 3), np.uint8
        )

        winner = "LEFT" if result.get("left", 0) > result.get("right", 0) else "RIGHT"

        cv2.putText(
            blank_frame,
            f"WINNER: {winner}",
            (blank_frame.shape[1] // 2 - 100, blank_frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0) if winner == "LEFT" else (0, 0, 255),
            3,
        )
        cv2.putText(
            blank_frame,
            f"LEFT: {result.get('left', 0)} - RIGHT: {result.get('right', 0)}",
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if winner == "LEFT" else (0, 0, 255),
            2,
        )

        self.new_frames += [blank_frame] * 30

        height, width, _ = self.new_frames[0].shape
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
        time_taken = {}
        start_time = time.time()
        print("Welcome to the Fish Detector! (currently detects blobs of orange)")
        self.getInput()
        if self.input_method == 0:
            self.vid = cv2.VideoCapture(0)
        elif self.input_method == 1:
            self.vid = cv2.VideoCapture(self.video_path)
        print("Reading frames...")
        time_taken["read"] = time.time()
        self.read_frames()
        time_taken["read"] = time.time() - time_taken["read"]
        print(f"Read {len(self.frames)} frames\nDetecting fish...")
        time_taken["detect"] = time.time()
        self.detect_fish()
        time_taken["detect"] = time.time() - time_taken["detect"]
        print(f"Detected {len(self.new_frames)} frames\nCreating video...")
        time_taken["create"] = time.time()
        self.create_video()
        time_taken["create"] = time.time() - time_taken["create"]
        print(
            f"Took {round((time.time() - start_time),2)} seconds to complete\nCleaning up..."
        )
        print("Video created!")
        print(f"Done! Result saved to {os.getcwd()}\output.mp4")
        print(
            "{} won!".format(
                "Left"
                if self.result.get("left", 0) > self.result.get("right", 0)
                else "Right"
            )
        )
        print(
            "Left: {} - Right: {}".format(
                self.result.get("left", 0), self.result.get("right", 0)
            )
        )
        print("Time taken:")
        for key, value in time_taken.items():
            print(f"{key.capitalize()}: {round(value,2)} seconds")
        print("Total time taken: {} seconds".format(round(sum(time_taken.values(),),2)))
        print("Most of the time was taken by", max(time_taken, key=time_taken.get))

if __name__ == "__main__":
    fd = FishDetector()
    fd.run()
