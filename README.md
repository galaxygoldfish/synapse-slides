# synapse-slides
This repository contains a companion Python script to use with a Muse EEG device that measures neural signals in real time and performs shortcut actions upon detection of certain actions, such as blinking. In the current state of the project, it allows for the advancement of a slideshow or fast-forwarding a video (right arrow key shortcut) when an intentional blink is detected. Read more below to learn more about the system and its' usage!

### System overview
<img width="100%" src="system-overview.svg"></img>

### Running the program
1. The prerequisite for this program is that you must have already connected your Muse EEG device to the computer you will be using to run the script
2. Clone this repository on your device, either using the command below in the terminal, or by downloading as a ZIP
   ```
   git clone https://galaxygoldfish/synapse-slides
   ```
3. Navigate to the folder containing the project files, and run the script using the following command
   ```
   python app.py
   ```
4. If you are intending to use this to advance a slideshow, the presentation app must be in focus

### Specifics during runtime
Currently, our script is not designed to run indefinitely. To customize how long to listen for EEG signals, open ```app.py``` and edit the ```RUNTIME_SECONDS``` variable (on line 45) to change how long (in seconds) the program will run
```python
# Record for 60 seconds / 1 minute
RUNTIME_SECONDS = 60
```

### Calibrating thresholds to improve accuracy
One limitation of the current state of this program is that it does not use adaptive learning models, so it is not likely to be as accurate when individuals use it without calibrating it. Here, we discuss the methods that are used by the program to detect a blink and how they can be edited for higher accuracy.

#### Visual data produced
You may notice that after running the program, when it exits, a few graph files are produced. These provide information about when the program thinks that a blink occurred.
1. ```time_freq_<time>.png``` will be a graph that looks something like this, where the top plot is a plot of the EEG signal in frequency domain, and the bottom plot is the raw signal in time domain
2. ```blink_vs_all_<time>.png``` will be a graph similar to the one above, but with blink times highlighted in red. The top is the EEG signal in frequency domain, middle is essentially a subset of the EEG signal where blinking is thought to occur in frequency domain, and the bottom graph is the EEG data in time domain with waves that surpass the threshold highloghted in red (assumed to be during blink times)

#### Calibrating in the code
Before we explain which parameters in the code to edit, first we will explain how a blink is detected.
