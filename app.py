from locale import normalize
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from scipy.signal import butter, lfilter, argrelextrema
import pyautogui

from utils import update_buffer, get_last_data
# Import functions from the utils module
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

# The length of the buffer in seconds to store EEG data
BUFFER_LENGTH = 5

# The length of each epoch in seconds to analyze
EPOCH_LENGTH = 1

# The amount of overlap between epochs in seconds
OVERLAP_LENGTH = 0

# The length of the shift between epochs in seconds
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
INDEX_CHANNEL = [0, 1]

# Time to record and respond to the neural signal
RUNTIME_SECONDS = 5

"""
Apply a Butterworth bandpass filter to the neural input.
This essentially filters out all frequencies that fall outside of the [low, high] range.
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
    
    # The unix timestamp when we start recording signals
    start = time.time()

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

    # Entire raw data set accumulator 
    entire_raw_data = np.array([])

    # Entire set of EEG voltage traces at blink times
    blink_data_all = np.array([])

    # Entire set of blink timestamps
    blink_time_all = np.array([])

    # TODO not sure if we still use this
    SEQUENCE_WINDOW = 5 #seconds or time interval for the triple blinks

    # TODO Local list that is reset at each time step holding timestamps of blink events
    blink_times = []
    last_slide_time = 0

    times_blinked = np.array([])
    magnitude_blinked = np.array([])

    collective_eeg_data = np.empty((0, 5))

    now_time = time.time()
    while now_time < start + RUNTIME_SECONDS:

        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        np_eeg_data = np.asarray(eeg_data)
        print(np_eeg_data)
        if (not(np_eeg_data.shape[0] == 0)):
            collective_eeg_data = np.append(collective_eeg_data, np_eeg_data, axis = 0)
            print(f"collective_eeg_data LOOP: {collective_eeg_data}")
            print(f"Loop running {now_time}", flush=True)

            # If no data is received or if the data are just zeros, continue to the next iteration
            if (len(eeg_data) == 0 or eeg_data[0] == [0, 0, 0, 0, 0]):
                print(f"time {now_time}", flush=True)
                continue 
            ''' check if the seond condiitonal actuall does something '''

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL[0]]
            # print(f'eeg_data: {eeg_data}')
            # print(f'channel data: {ch_data}')
            ''' no need for this line if we want data from all channels '''

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)
            ''' isn't eeg_buffer input just a list of zeros? should check this. this would
                mean that eeg_buffer output is just the list of zeros + ch_data '''

            # Get newest samples from the buffer
            epoch_data = get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
            epoch_timestamps = np.linspace(start, now_time, len(epoch_data))
            # print(f'epoch_data: {epoch_data}')
            # print(f'time: {epoch_timestamps} + now_time, epoch_data, fs {now_time} {epoch_data} {fs}')

            # Blink detection: look for large spikes in the most recent data
            # Use the absolute value to detect both up and down spikes
            
            # The filtered epoch is the epoch data with non-blink frequencies removed
            filtered_epoch = bandpass(epoch_data, fs)
            filtered_epoch_abs = np.abs(filtered_epoch)
            filtered_epoch_np = np.array(filtered_epoch)

            MAX_AMP = np.max(filtered_epoch)
            MAX_AMP_index = np.where(epoch_data == MAX_AMP)
            # print(MAX_AMP)
            # print(MAX_AMP_index)

            # Append the filtered epoch data to the whole data set
            if len(entire_raw_data) == 0:
                entire_raw_data = filtered_epoch_np
                whole_timestamps = epoch_timestamps
            else:
                entire_raw_data = np.append(entire_raw_data, filtered_epoch_np)
                whole_timestamps = np.append(whole_timestamps, epoch_timestamps)

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

                        # # blink_times = [i for i in blink_times if now_time - i <= SEQUENCE_WINDOW]
                        # print(f'blink_times: {blink_times}')
                        # print(f'times_blinked: {times_blinked}')
                        # print(magnitude_blinked)
                        # print(epoch_timestamps)
                        # # print(epoch_timestamps[MAX_AMP_index])
                        # print()
                        # # times_blinked = np.append(times_blinked, epoch_timestamps[MAX_AMP_index])
                        # # magnitude_blinked = np.append(magnitude_blinked, MAX_AMP)

                        # If the number of blinks aligns with our SEQUENCE_COUNT, we call the sequence blink action               
                        if len(blink_times) >= SEQUENCE_COUNT:
                            sequence_blink_action()
                            blink_times.clear()
                        # Otherwise, it is classified as a single blink action
                        else:
                            single_blink_action()

                # print(f"Timestamp : {now_time}")
                # print(f"Blinks: {blink_count}")

                # Blink mask is a boolean array that indicates which samples are considered blinks
                filtered_epoch_flattened = bandpass(epoch_data, fs).flatten()
                blink_mask = (np.abs(filtered_epoch_flattened) > BLINK_MIN_THRESHOLD) & \
                            (np.abs(filtered_epoch_flattened) < BLINK_MAX_THRESHOLD)

                # Check if there are any blinks in the current epoch
                # If so, append the blink data to the blink data accumulator
                if blink_mask.any():
                    blink_data_all = np.append(blink_data_all, filtered_epoch_flattened[blink_mask])
                    blink_time_all = np.append(blink_time_all, epoch_timestamps[blink_mask])

        # The current unix timestamp for this recording step
        now_time = time.time()


    # If the runtime reaches the desired maximum number of seconds

    # Time to frequency domain converstion
    print (f'Entire raw data: {entire_raw_data.shape}')
    freqs, mag = time_to_frequency_domain(entire_raw_data, fs)
    
    # Create the plot with the raw data (not filtered for blinks)
    plot_entire_timeANDfreq_data(freqs, mag, entire_raw_data, whole_timestamps, now_time)
    

    # Create plot of frequency domain for raw and only during blink times,
    # and show in the raw EEG data where we predict blinks to be happening, highlighted in red
    # plot_blink_selected_data(blink_data_all, fs, freqs, normalized_mag, whole_timestamps, entire_raw_data, now_time, blink_time_all)

    # Export raw data as csv (may remove later)
    # export_raw_csv(blink_time_all, blink_data_all, whole_timestamps, entire_raw_data, now_time)
    
    # Stop the entire program and stop recording signals

    print(f'collective_eeg_data: {collective_eeg_data.shape}')

    time_arr, abs = filter_data(collective_eeg_data, fs, epoch_data)

    num_signals, signal_length = collective_eeg_data.T.shape
    fft_length = signal_length // 2 + 1
    
    freq_arr_2d = np.zeros((num_signals, fft_length))
    mag_arr_2d = np.zeros((num_signals, fft_length))
    transposed_collective_eeg_data = collective_eeg_data.T
    for i in range(transposed_collective_eeg_data.shape[0]):
        f, m = time_to_frequency_domain(transposed_collective_eeg_data[i], fs)
        freq_arr_2d[i] = f
        mag_arr_2d[i] = m

    print(f'freq_arr_2d: {freq_arr_2d.shape}')

    plt.figure(figsize=(10, 6))
    # First subplot (top)
    plt.subplot(2, 1, 1)
    labels = [f"Datastream {i+1}" for i in range(collective_eeg_data.shape[1])]
    plt.plot(time_arr, label = labels)
    plt.title("Time Domain EEG Data")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (µV)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freq_arr_2d.T, mag_arr_2d.T, label = labels)
    plt.title("Frequency Domain EEG Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.savefig(f"time_&_freq_domain{now_time}.png")


    print(collective_eeg_data)

'''
This function filters the collective EEG data to remove non-blink frequencies
Parameters:
    collective_eeg_data: The EEG data collected from the Muse device
    fs: Sampling frequency in HZ
    epoch_data: The epoch data to filter
Returns:
    new_arr: The filtered EEG data with non-blink frequencies removed
'''
def filter_data(collective_eeg_data, fs, epoch_data):

    # The filtered epoch is the epoch data with non-blink frequencies removed
    print(collective_eeg_data[:, 1])
    transposed_data = collective_eeg_data.T
    tr_bandpassed_data = np.zeros_like(transposed_data)
    for i in range(transposed_data.shape[0]):
        tr_bandpassed_data[i] = bandpass(transposed_data[i], fs)
    bandpassed_data = tr_bandpassed_data.T
    print(f'filtered_epoch_1: {bandpassed_data}')
    filtered_epoch_np = np.array(bandpassed_data)

    # Append the filtered epoch data to the whole data set
    new_arr = np.array([])

    if len(new_arr) == 0:
        new_arr = filtered_epoch_np
    else:
        new_arr = np.concatenate((collective_eeg_data, filtered_epoch_np))

    return new_arr, np.abs(bandpassed_data) #abs

"""
Create the plot of the raw EEG data on the bottom subplot and the frequency domain
on the top subplot

Parameters:
    freqs: The frequencies of the FFT
    normalized_mag: The normalized magnitude of the FFT
    entire_raw_data: The entire neural recording, raw
    whole_timestamps: The timestamps of the entire neural recording
    now_time: The current unix timestamp
    mag: The magnitude of the FFT

The plot will also be saved as a PNG file named time_freq_<now_time>.png
"""
def plot_entire_timeANDfreq_data(freqs, mag, entire_raw_data, whole_timestamps, now_time):
    plt.figure(figsize=(10, 6))

    # First subplot (top)
    plt.subplot(2, 1, 1)
    plt.plot(entire_raw_data)
    plt.title("Live EEG Data Epoch")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (µV)")

    # df_freq = pd.DataFrame({'frequencies': freqs, 'magnitude': mag})
    # df_freq.to_csv('freq_data_with_timestamps_{int(now_time)}.csv', index=False)

    # Second subplot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(freqs, mag)
    plt.title("Live EEG Data Epoch")
    plt.xlim([0, 50])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.savefig(f"time_freq_{int(now_time)}.png")

"""
Create a plot of the frequency domain for all samples, blink-only samples, and plot the raw
EEG data with the classified blink times highlighted in red

Parameters:
    blink_data_all: The EEG data at the blink timestamps
    fs: Sampling frequency in HZ
    freqs: The frequencies of the FFT
    normalized_mag: The normalized magnitude of the FFT
    whole_timestamps: The timestamps of the entire neural recording
    entire_raw_data: The entire neural recording, raw
    now_time: The current unix timestamp
    blink_time_all: The timestamps of the blinks

This will also save the figure as a PNG file named blink_vs_all_<now_time>.png
"""
def plot_blink_selected_data(blink_data_all, fs, freqs, normalized_mag, whole_timestamps, entire_raw_data, now_time, blink_time_all):
    n_blink = len(blink_data_all)
    if n_blink > 0:
        fft_blink  = np.fft.rfft(blink_data_all)
        mag_blink  = np.abs(fft_blink) / n_blink
        freqs_blnk = np.fft.rfftfreq(n_blink, d=1/fs)
    else: # no blink samples collected
        mag_blink, freqs_blnk = np.array([0]), np.array([0])
    # Normalize the blink magnitude
    mag_blink_max = np.max(mag_blink)
    normalized_mag_blink = mag_blink/mag_blink_max
    plt.figure(figsize=(11, 7))
    # All samples frequency domain
    plt.subplot(3, 1, 1)
    plt.plot(freqs, normalized_mag)
    plt.title('All Samples – Frequency Domain')
    plt.ylabel('Magnitude (µV)')
    plt.xlim(0, 10)
    # Blink only sample frequency domain
    plt.subplot(3, 1, 2)
    plt.plot(freqs_blnk, normalized_mag_blink, color='crimson')
    plt.title('Blink Samples – Frequency Domain')
    plt.ylabel('Magnitude (µV)')
    plt.xlim(0, 10)
    # Raw data with blink samples highlighted
    plt.subplot(3, 1, 3)
    plt.plot(whole_timestamps, entire_raw_data, label='Full stream')
    plt.scatter(blink_time_all, blink_data_all, color='crimson', linewidth=2, label='Blink samples')
    # Make a purple scatter plot with only the local max and mins of the blink_data_all
    
    if len(blink_data_all) > 2:
        max_indices = argrelextrema(blink_data_all, np.greater)[0]
        min_indices = argrelextrema(blink_data_all, np.less)[0]
        # Combine and sort indices
        extrema_indices = np.sort(np.concatenate((max_indices, min_indices)))
        # Plot local maxima and minima as purple points
        plt.scatter(
            np.array(blink_time_all)[extrema_indices],
            np.array(blink_data_all)[extrema_indices],
            color='purple', linewidth=2, label='Blink local max/min'
        )

    plt.title('Time-Series with Blink Samples Marked')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (µV)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'blink_vs_all_{int(now_time)}.png')

"""
Saves the raw EEG data to a CSV file with two columns (timestamp, voltage)

Parameters:
    blink_time_all: The timestamps of the blinks
    blink_data_all: The EEG data at the blink timestamps
    whole_timestamps: The timestamps of the entire neural recording
    entire_raw_data: The entire neural recording, raw
    now_time: The current unix timestamp
"""
def export_raw_csv(blink_time_all, blink_data_all, whole_timestamps, entire_raw_data, now_time):
    df_time = pd.DataFrame({'timestamp': whole_timestamps, 'eeg_value': entire_raw_data})
    df_time.to_csv(f'time_data_with_timestamps_{int(now_time)}.csv', index=False)
    df_blink = pd.DataFrame({'timestamp': blink_time_all,'eeg_value': blink_data_all})
    df_blink.to_csv(f'blink_only_time_series_{int(now_time)}.csv', index=False)
            

def time_to_frequency_domain(signal, fs):
    n = len(signal)
    magnitude = np.abs(np.fft.rfft(signal)) / n
    normalized_magnitude = magnitude / np.max(magnitude)
    freqs = np.fft.rfftfreq(n, 1/fs)

    return freqs, normalized_magnitude

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

if __name__ == '__main__':
    record_live()