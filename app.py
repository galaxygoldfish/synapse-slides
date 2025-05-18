import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import threading
from scipy.signal import butter, lfilter
import pyautogui

# Import functions from the utils module
from utils import update_buffer, get_last_data, compute_band_powers
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [1]
                 
# Eitan: low=4.0 high=13.0 BLINK_THRESHOLD=180 MIN_BLINK_INTERVAL=3  for motion
# Max: low = .1, high = 3.0 Blink in mV = 29 for light to normal blinking
# Max : low = .1, high = 3.0; max mV = 200; min mV = 42 for light blinking and movement
# Omor: low= .1, high = 5.0; Theshold = 29 mV for blinding Med to hard 


def bandpass(data, fs, low= 0.1 , high= 3.0 ,order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=0)
        
def record_live():
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

    last_blink_time = 0
    blink_count = 0
    
    BLINK_THRESHOLD =  43 #  microvolts (adjust this based on your signal)
    MAX_THRESHOLD = 200
    MIN_BLINK_INTERVAL =  1 # seconds (minimum interval between blinks)


    #media resume and pause variables 
    SEQUENCE_WINDOW = 3  #seconds or time interval for the thripple blinks
    SEQUENCE_COUNT = 3  #how many times Max have to blink under the window
    blink_times = []
    last_slide_time = 0


    
    start = time.time()
    whole_data = np.array([])
    whole_data = np.array([])

    while True:
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
        data_epoch = get_last_data(eeg_buffer,
                                   EPOCH_LENGTH * fs)
        epoch_timestamps = np.linspace(now_time - (len(data_epoch) / fs), now_time, len(data_epoch))

        # Blink detection: look for large spikes in the most recent data
        # Use the absolute value to detect both up and down spikes
        filtered_epoch = bandpass(data_epoch, fs)
        if np.max(np.abs(filtered_epoch)) > BLINK_THRESHOLD and np.max(np.abs(filtered_epoch)) < MAX_THRESHOLD:
            
            if now_time - last_blink_time > MIN_BLINK_INTERVAL:
                print( "Blink detected!")
                pyautogui.press("right")
                blink_count += 1
                last_blink_time = now_time
                blink_times.append(now_time)
                blink_times = [t for t in blink_times if now_time - t <= SEQUENCE_WINDOW]

                if len(blink_times) >= SEQUENCE_COUNT:
                    print("Triple blink detected: toggling play/pause")
                    pyautogui.press("playpause")  # toggles media play/pause
                    blink_times.clear()
                    last_slide_time = now_time
                elif now_time - last_slide_time > MIN_BLINK_INTERVAL:
                    print("Single blink detected: advancing slide")
                    pyautogui.press("right")  # goes to next slide
                    last_slide_time = now_time

        band_powers = compute_band_powers(data_epoch, fs)
        delta, theta, alpha, beta = band_powers

        print(f"Timestamp : {now_time}")
        #print(f"Blinks: {blink_count}")

        

        # Compute band powers
        band_powers = compute_band_powers(data_epoch, fs)
        delta, theta, alpha, beta = band_powers
        
        filtered_epoch_np = np.array(filtered_epoch)
        if len(whole_data) == 0:
            whole_data = filtered_epoch_np
            whole_timestamps = epoch_timestamps
        else:
            whole_data = np.append(whole_data, filtered_epoch_np)
            whole_timestamps = np.append(whole_timestamps, epoch_timestamps)

        # --- Add this block for live plotting ---
        # plt.ion()  # Turn on interactive mode
        # plt.clf()  # Clear the current figure
        # plt.plot(data_epoch)
        # plt.title("Live EEG Data Epoch")
        # plt.xlabel("Sample")
        # plt.ylabel("Amplitude")
        # plt.savefig(f"final_data_epoch{time.time()}.png")
        # plt.pause(0.001)

        if time.time() > start + 60:
            # Time to frequency domain converstion (FFT)
            n = len(whole_data)
            fft_vals = np.fft.rfft(whole_data)
            mag = np.abs(fft_vals) / n
            freqs = np.fft.rfftfreq(n, d=1/260)
            
            plt.figure(figsize=(10, 6))

            # First subplot (top)
            plt.subplot(2, 1, 1)
            plt.plot(freqs, mag)
            plt.title("Frequency Domain")
            plt.xlabel("Frequency Hz")
            plt.xlim([0, 10])
            plt.ylabel("Magnitude")
            # plt.savefig(f"FINAL_freqs{now_time}.png")
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

            plt.tight_layout()
            break

    
    
            
def record(duration_seconds=10, output_file='eeg_bandpowers.csv'):
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

    # Initialize storage for band power time series
    data_log = []

    print(f'Recording for {duration_seconds} seconds...')

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
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
        data_epoch = get_last_data(eeg_buffer,
                                   EPOCH_LENGTH * fs)

        # Compute band powers
        band_powers = compute_band_powers(data_epoch, fs)
        delta, theta, alpha, beta = band_powers
        timestamp_now = time.time() - start_time

        # Log band powers with timestamp
        data_log.append({
            'time': timestamp_now,
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'delta': delta
        })

    df = pd.DataFrame(data_log)
    df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')

    return df

if __name__ == '__main__':
    record_live()