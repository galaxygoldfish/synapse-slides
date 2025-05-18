import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from scipy.signal import butter, lfilter
import pyautogui

from utils import update_buffer, get_last_data
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

# TODO figure these out and comment
BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH


# The min and max thresholds here are the range of neural signal we consider to be a blink
# These parameters must be relatively tuned to the user of the device
# Here, they have been tuned to work for Max

# The lower limit of the blink threshold, in µV
BLINK_MIN_THRESHOLD = 250
# The upper limit of the blink threshold, in µV
BLINK_MAX_THRESHOLD = 1200

# Number of blinks that are required to trigger the sequence callback
SEQUENCE_COUNT =  2

# Minimum interval between blinks, in seconds
MIN_BLINK_INTERVAL =  4

# This is the channel from Muse that we want to record from
INDEX_CHANNEL = [1]

# Time to record and respond to the neural signal
RUNTIME_SECONDS = 60

"""
Apply a Butterworth bandpass filter to the neural input.
This essentially filters out all frequencies that fall outside of the [low, high] range
In our case, we use this to filter out signals that might be unrelated to blinking

Parameters:
    data: The input EEG signal
    fs: Sampling frequency in HZ
    low: The low cutoff frequency
    high: The high cutoff frequency
    order: Order of the Butterworth filter

Returns:
    The filtered signal with frequencies outside [low, high] HZ removed
"""
def bandpass(data, fs, low= 0.1 , high= 5.5 ,order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=0)
        
"""
Attempt to connect to the Muse EEG device and throw an error if not found

Return: The StreamInlet that provides live EEG data
"""
def connect_eeg():

    # Search for active LSL streams and connect to Muse
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)

    return inlet

"""
Records live neural signal from Muse EEG and listens for blink actions

One blink: Right arrow key is pressed (advance slides)
Two blinks: (Within a set time window) Play/pause key is pressed
"""
def record_live():
    
    # Receive data stream from Muse
    inlet = connect_eeg()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    
    # For use with the notch filter
    filter_state = None

    # The time (unix) to record the difference between blinks
    last_blink_time = 0

    # The number of total blinks in the session
    blink_count = 0
    
    # The unix timestamp when we start recording signals
    start = time.time()

    # Entire raw data set accumulator 
    whole_data = np.array([])

    # Entire set of EEG voltage traces at blink times
    blink_data_all = np.array([])

    # Entire set of blink timestamps
    blink_time_all = np.array([]) 

    # TODO not sure if we still use this
    SEQUENCE_WINDOW = 5 #seconds or time interval for the thripple blinks

    # Local list that is reset at each time step holding timestamps of blink events
    blink_times = []

    while True:

        # The current unix timestamp for this recording step
        now_time = time.time()

        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        if len(eeg_data) == 0:
            continue

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

        # Update EEG buffer with the new data
        eeg_buffer, filter_state = update_buffer(
            eeg_buffer, ch_data, notch=True,
            filter_state=filter_state)

        # Get newest samples from the buffer
        data_epoch = get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
        epoch_timestamps = np.linspace(now_time - (len(data_epoch) / fs), now_time, len(data_epoch))

        # Blink detection: look for large spikes in the most recent data
        # Use the absolute value to detect both up and down spikes
        filtered_epoch = bandpass(data_epoch, fs)
        filtered_epoch_abs = np.abs(filtered_epoch)

        # If the local max of the epoch data falls within our defined threshold and max, we consider it a blink
        if np.max(filtered_epoch_abs) > BLINK_MIN_THRESHOLD and np.max(filtered_epoch_abs) < BLINK_MAX_THRESHOLD:
           
           # To filter out extraneous blinks, false positives, we check if the MIN_BLINK_INTERVAL
           # has passed since the last blink, so that there is a cooldown period between actions
           if now_time - last_blink_time > MIN_BLINK_INTERVAL:
                
                # Record the blink
                print("Blink detected!")
                blink_count += 1
                
                # Append to our local list of blink times in the current step
                last_blink_time = now_time
                blink_times.append(now_time)

                # blink_times = [i for i in blink_times if now_time - i <= SEQUENCE_WINDOW]
                # print(blink_times)

                # If the number of blinks aligns with our SEQUENCE_COUNT, we call the sequence blink action               
                if len(blink_times) >= SEQUENCE_COUNT:
                    sequence_blink_action()
                    blink_times.clear()
                # Otherwise, it is classified as a single blink action
                else:
                    single_blink_action()

        # print(f"Timestamp : {now_time}")
        print(f"Blinks: {blink_count}")

        # STILL NEED TO COMMENT AND FIGURE OUT LOGIC HERE -----

        filtered_epoch_np = np.array(filtered_epoch)

        if len(whole_data) == 0:
            whole_data = filtered_epoch_np
            whole_timestamps = epoch_timestamps
        else:
            whole_data = np.append(whole_data, filtered_epoch_np)
            whole_timestamps = np.append(whole_timestamps, epoch_timestamps)

        filtered_epoch_flattened = bandpass(data_epoch, fs).flatten()
        blink_mask = (np.abs(filtered_epoch_flattened) > BLINK_MIN_THRESHOLD) & (np.abs(filtered_epoch_flattened) < BLINK_MAX_THRESHOLD)

        if blink_mask.any():
            blink_data_all = np.append(blink_data_all, filtered_epoch_flattened[blink_mask])
            blink_time_all = np.append(blink_time_all, epoch_timestamps[blink_mask])

        # ------------------------------------------------------

        # Once the runtime reaches the desired maximum number of seconds
        if time.time() > start + RUNTIME_SECONDS:

            # Time to frequency domain converstion
            freqs, normalized_mag, mag = time_to_frequency_domain(whole_data)
            
            # Create the plot with the raw data (not filtered for blinks)
            plot_entire_data(freqs, normalized_mag, whole_data, whole_timestamps, now_time, mag)
     
            # Create plot of frequency domain for raw and only during blink times,
            # and show in the raw EEG data where we predict blinks to be happening, highlighted in red
            plot_blink_selected_data(blink_data_all, fs, freqs, normalized_mag, whole_timestamps, whole_data, now_time, blink_time_all)

            # Export raw data as csv (may remove later)
            export_raw_csv(blink_time_all, blink_data_all)
            
            # Stop the entire program and stop recording signals
            break


"""
Create the plot of the raw EEG data on the bottom subplot and the frequency domain
on the top subplot

Parameters:
    TODO

TODO explain file naming
"""
def plot_entire_data(freqs, normalized_mag, whole_data, whole_timestamps, now_time, mag):
    plt.figure(figsize=(10, 6))
    # First subplot (top)
    plt.subplot(2, 1, 1)
    plt.plot(freqs, normalized_mag)
    plt.title("Frequency Domain")
    plt.xlabel("Frequency Hz")
    plt.xlim([0, 10])
    plt.ylabel("Magnitude")
    df_freq = pd.DataFrame({'frequencies': freqs, 'magnitude': mag})
    df_freq.to_csv('freq_data_with_timestamps.csv', index=False)
    # Second subplot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(whole_data)
    plt.title("Live EEG Data Epoch")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(f"FINAL_time&freq{now_time}.png")
    df_time = pd.DataFrame({'timestamp': whole_timestamps, 'eeg_value': whole_data})
    df_time.to_csv('time_data_with_timestamps.csv', index=False)

"""
TODO comment
"""
def plot_blink_selected_data(blink_data_all, fs, freqs, normalized_mag, whole_timestamps, whole_data, now_time, blink_time_all):
    n_blink = len(blink_data_all)
    if n_blink > 0:
        fft_blink  = np.fft.rfft(blink_data_all)
        mag_blink  = np.abs(fft_blink) / n_blink
        freqs_blnk = np.fft.rfftfreq(n_blink, d=1/fs)
    else:                                      # no blink samples collected
        mag_blink, freqs_blnk = np.array([0]), np.array([0])

    mag_blink_max = np.max(mag_blink)
    normalized_mag_blink = mag_blink/mag_blink_max

    plt.figure(figsize=(11, 7))

    plt.subplot(3, 1, 1)
    plt.plot(freqs, normalized_mag)
    plt.title('All Samples – Frequency Domain')
    plt.ylabel('Magnitude (µV)')
    plt.xlim(0, 10)

    plt.subplot(3, 1, 2)
    plt.plot(freqs_blnk, normalized_mag_blink, color='crimson')
    plt.title('BLINK-ONLY Samples – Frequency Domain')
    plt.ylabel('Magnitude (µV)')
    plt.xlim(0, 10)

    plt.subplot(3, 1, 3)
    plt.plot(whole_timestamps, whole_data, label='Full stream')
    plt.scatter(blink_time_all, blink_data_all, color='crimson', linewidth=2, label='Blink samples')
    plt.title('Time-Series with Blink Samples Marked')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (µV)')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'blink_vs_all_{int(now_time)}.png')
    plt.show()

"""
TODO comment
"""
def export_raw_csv(blink_time_all, blink_data_all):
    pd.DataFrame({
        't': blink_time_all,
        'eeg': blink_data_all
    }).to_csv('blink_only_time_series.csv', index=False)
            

"""
Convert data from a time-based domain to a frequency-based domain using
a Fast Fourier Transform (FFT)

Parameters:
    whole_data: The entire neural recording, raw

Returns:
    TODO
"""
def time_to_frequency_domain(whole_data):
    n = len(whole_data)
    fft_vals = np.fft.rfft(whole_data)
    mag = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, d = 1 / 260)
    mag_max = np.max(mag)
    normalized_mag = mag/mag_max
    return normalized_mag, freqs, mag

"""
Callback to be invoked when a single blink action is detected

In this case, the right arrow key is pressed
This is useful if you are focused on a slideshow window for example, to move next slide
We have also found that this works to fast-forward in YouTube
"""
def single_blink_action():
    print("Single blink detected")
    pyautogui.press("right")

"""
Callback to be invoked when the SEQUENCE_COUNT number of blinks are detected

In this case, the play/pause keyboard shortcut is pressed
This allows for global media control (videos, music, etc.)
"""
def sequence_blink_action():
    print("Double blink detected")
    pyautogui.press("mediaplaypause")

# Main function to run when the script is executed
if __name__ == '__main__':
    record_live()