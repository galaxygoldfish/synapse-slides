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

# Eitan: low=4.0 high=13.0 BLINK_THRESHOLD=180 MIN_BLINK_INTERVAL=3

def bandpass(data, fs, low= 4.0 , high= 6.0 ,order=4):
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
    
    BLINK_THRESHOLD = 120 # microvolts (adjust this based on your signal)
    MIN_BLINK_INTERVAL =  2    # seconds (minimum interval between blinks)
    SEQUENCE_WINDOW    = 3         # seconds in which to count triple‐blinks
    SEQUENCE_COUNT     = 3         # number of blinks needed in that window
    MIN_SLIDE_INTERVAL = 25        # seconds between single‐blink slides
        

    while True:
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
        
        # Blink detection: look for large spikes in the most recent data
        # Use the absolute value to detect both up and down spikes
        filtered_epoch = bandpass(data_epoch, fs)
        if np.max(np.abs(filtered_epoch)) > BLINK_THRESHOLD:
            now = time.time()


            # video or media resume or pauseing code: 
            # 1) append this blink timestamp
            blink_times.append(now)
            # 2) purge any blinks older than SEQUENCE_WINDOW
            blink_times = [t for t in blink_times if now - t <= SEQUENCE_WINDOW]

            # 3) if we have SEQUENCE_COUNT blinks in the window → play/pause
            if len(blink_times) >= SEQUENCE_COUNT:
                print("Triple‐blink detected: toggling play/pause")
                pyautogui.press("playpause")       # Windows media key
                blink_times.clear()                 # reset
                last_slide_time = now               # avoid immediate slide
            # 4) else if enough time passed since last slide → next slide
            elif now - last_slide_time > MIN_SLIDE_INTERVAL:
                print("Single blink: advancing slide")
                pyautogui.press("right")
                last_slide_time = now
            #ending of the media code


            if now - last_blink_time > MIN_BLINK_INTERVAL:
                print("Blink detected!")
                pyautogui.press("right")
                blink_count += 1
                last_blink_time = now

        band_powers = compute_band_powers(data_epoch, fs)
        delta, theta, alpha, beta = band_powers

        print(f"Blinks: {blink_count}")

        # Compute band powers
        band_powers = compute_band_powers(data_epoch, fs)
        delta, theta, alpha, beta = band_powers
        
        #rint(data_epoch)
            
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


