# GazeType: Adaptive Eye-Gaze Communication Keyboard

GazeType is a webcam-based assistive communication prototype. It lets a user choose characters and common words by looking toward on-screen controls and confirming selections with deliberate blinks.

## Features

- Real-time face and eye landmark tracking through MediaPipe Face Landmarker
- Left and right gaze selection for the alphabet keyboard
- Blink calibration with the option to save and reuse the calibration profile
- Two-blink selection: blink once to pause a key and again to confirm it
- Adaptive keyboard scanning speed
- Word prediction suggestions from a compact offline word list
- Letter, space, backspace, prediction, and cancel controls
- Session performance metrics, including selections, corrections, rejected blinks, and estimated typing speed
- Single dashboard interface with camera tracking, keyboard, text, status, and guidance
- Audio feedback for keyboard and key selection

## Requirements

- Python 3.10 or newer
- A webcam and a display
- The dependencies listed in `requirements.txt`

Install dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Run

Run the application from this directory so its model and sound assets are found:

```bash
python GazeType_Final_Project.py
```

Press `R` to recalibrate and `Esc` to exit.

## How to use

1. Keep your face visible to the webcam. If calibration is required, keep your eyes open for the first stage and blink normally three times during the next stage.
2. In the keyboard selection screen, look left or right and hold your gaze to open that keyboard.
3. Watch the highlighted key as the keyboard scans through its choices.
4. Blink once to pause on a key, then blink again within the confirmation period to select it.
5. Use `SPACE` to add a space, `BACK` to delete the last character, and `PRED` to open available word suggestions.
6. Use gaze and blink selection to navigate the prediction choices; select a suggestion or cancel to return to typing.

## Keyboard layouts

| Left keyboard | Right keyboard |
| --- | --- |
| Q W E R T | Y U I O P |
| A S D F G | H J K L SPACE |
| Z X C V BACK | B N M PRED BACK |

When prediction controls are active, a small prediction keyboard is shown to choose among suggested words or cancel.

## Packages and technologies

| Package / asset | Purpose |
| --- | --- |
| Python | Application logic and interaction state |
| OpenCV (`opencv-python`) | Webcam capture, image drawing, dashboard display, and keyboard input |
| NumPy | Image arrays and numerical smoothing |
| MediaPipe | Face and eye landmark detection through the Tasks Vision Face Landmarker |
| Pyglet | Playback of WAV feedback sounds |
| Python standard library | JSON profile and session storage, timing, rolling buffers, and calculations |
| `face_landmarker.task` | MediaPipe model asset loaded at runtime |

## Project files

| File | Purpose |
| --- | --- |
| `GazeType_Final_Project.py` | Main application |
| `face_landmarker.task` | Face landmark model used by MediaPipe |
| `sound.wav`, `left.wav`, `right.wav` | Key selection and keyboard direction sounds |
| `Adaptive-Eye-Gaze-Communication-Keyboard.pptx` | Project presentation |
| `GazeType_Final_Project_Report.docx` | Detailed project report |
| `DA2_Result_Tabulation_Keyboard_Typing_with_Eyes.docx` | Functional result tabulation |
| `DEMO.mp4` | Demonstration video; stored with Git LFS because it is larger than 100 MiB |

The application creates `gazetype_profile.json` to store calibration values and `gazetype_session_history.json` to store session metrics. These are local runtime data files and are excluded from version control by `.gitignore`.

## Settings

The main configuration constants are near the top of `GazeType_Final_Project.py`. They include the camera index, gaze thresholds, calibration duration, blink smoothing, confirmation timeout, and minimum, maximum, and base scan speeds.

## Troubleshooting

### Camera fails to open

Check that the webcam is connected and not being used by another application. Change `CAMERA_INDEX` if the desired camera is not device `0`.

### Model or audio asset cannot be found

Run the script from this directory and confirm that `face_landmarker.task` and the WAV files are present beside it.

### Gaze or blink selection is inconsistent

Face the camera in steady lighting, keep your face within the frame, and complete calibration while seated at your normal distance from the camera.

### The demo video is not downloaded after cloning

Install Git LFS, then clone the repository. GitHub stores the video as an LFS object; without Git LFS a clone contains only a small pointer file in place of the video.

## Scope

GazeType displays composed text in its application interface and records local session metrics. It does not send keystrokes to other desktop applications.

## License

No license has been specified for this project.
