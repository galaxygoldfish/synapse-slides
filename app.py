import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from scipy.signal import butter, lfilter
import pyautogui

# Import functions from the utils module
from utils import epoch, update_buffer, get_last_data
from pylsl import StreamInlet, resolve_byprop

app = Flask(__name__)

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# This is the channel from Muse that we want to record from
INDEX_CHANNEL = [1]

# Eitan: low=4.0 high=13.0 BLINK_THRESHOLD=180 MIN_BLINK_INTERVAL=3  for motion
# Max: low = .1, high = 3.0 Blink in mV = 29 for light to normal blinking
# Max : low = .1, high = 3.0; max mV = 200; min mV = 42 for light blinking and movement
# Omor: low= .1, high = 5.0; Theshold = 29 mV for blinding Med to hard 

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
Records live neural signal from Muse EEG

"""
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
    
    BLINK_THRESHOLD = 250 #  microvolts (adjust this based on your signal)
    MAX_THRESHOLD = 1200
    MIN_BLINK_INTERVAL =  4 # seconds (minimum interval between blinks)

    
    start = time.time()
    whole_data = np.array([])
    whole_data = np.array([])

    blink_data_all = np.array([])
    blink_time_all = np.array([]) 


    SEQUENCE_WINDOW = 5 #seconds or time interval for the thripple blinks
    SEQUENCE_COUNT =  2 #how many times Max have to blink under the window
    blink_times = []
    last_slide_time = 0

    times_blinked = np.array([])
    magnitude_blinked = np.array([])

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

        filtered_epoch_abs = np.abs(filtered_epoch)

        MAX_AMP = np.max(filtered_epoch)
        MAX_AMP_index = np.where(data_epoch == MAX_AMP)
        print(MAX_AMP)
        print(MAX_AMP_index)

        if np.max(filtered_epoch_abs) > BLINK_THRESHOLD and np.max(filtered_epoch_abs) < MAX_THRESHOLD:
            
           if now_time - last_blink_time > MIN_BLINK_INTERVAL:
                print( "Blink detected!")
                blink_count += 1
                last_blink_time = now_time
                blink_times.append(now_time)
                # blink_times = [i for i in blink_times if now_time - i <= SEQUENCE_WINDOW]
                print(blink_times)

                print(times_blinked)
                print(magnitude_blinked)
                print(epoch_timestamps)
                print(epoch_timestamps[MAX_AMP_index])
                print()
                times_blinked = np.append(times_blinked, epoch_timestamps[MAX_AMP_index])
                magnitude_blinked = np.append(magnitude_blinked, MAX_AMP)

                if len(blink_times) >= SEQUENCE_COUNT:
                    print("Triple blink detected: toggling play/pause")
                    pyautogui.press("playpause")  # toggles media play/pause
                    blink_times.clear()
                    last_slide_time = now_time
                else:
                    print("Single blink detected")
                    pyautogui.press("right")
                    #blink_times.clear()
                    last_slide_time = now_time

        # print(f"Timestamp : {now_time}")
        print(f"Blinks: {blink_count}")

        filtered_epoch_np = np.array(filtered_epoch)
        if len(whole_data) == 0:
            whole_data = filtered_epoch_np
            whole_timestamps = epoch_timestamps
        else:
            whole_data = np.append(whole_data, filtered_epoch_np)
            whole_timestamps = np.append(whole_timestamps, epoch_timestamps)

        # blink threshold = 250
        # max threshold = 1200
        # ...existing code...
        filtered_epoch_flattened = bandpass(data_epoch, fs).flatten()
        # ...existing code...
        blink_mask = (np.abs(filtered_epoch_flattened) > 250) & (np.abs(filtered_epoch_flattened) < 1200)

        if blink_mask.any():
            blink_data_all = np.append(blink_data_all, filtered_epoch_flattened[blink_mask])
            blink_time_all = np.append(blink_time_all, epoch_timestamps[blink_mask])
        # ...existing code...
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
            mag_max = np.max(mag)
            normalized_mag = mag/mag_max
            
            plt.figure(figsize=(10, 6))

            # First subplot (top)
            plt.subplot(2, 1, 1)
            plt.plot(freqs, normalized_mag)
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

            # ---------- blink-only spectrum ----------
            n_blink = len(blink_data_all)
            if n_blink > 0:
                fft_blink  = np.fft.rfft(blink_data_all)
                mag_blink  = np.abs(fft_blink) / n_blink
                freqs_blnk = np.fft.rfftfreq(n_blink, d=1/fs)
            else:                                      # no blink samples collected
                mag_blink, freqs_blnk = np.array([0]), np.array([0])

            mag_blink_max = np.max(mag_blink)
            normalized_mag_blink = mag_blink/mag_blink_max

            # ---------- combined figure ----------
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
            plt.scatter(times_blinked, magnitude_blinked, color = 'blue', linewidth = 2, label = 'Counted Blinks')
            plt.title('Time-Series with Blink Samples Marked')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude (µV)')
            plt.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(f'blink_vs_all_{int(now_time)}.png')
            plt.show()

            # -------- optional CSV dump -----------
            pd.DataFrame({'t'  : blink_time_all,
                          'eeg': blink_data_all}).to_csv(
                          'blink_only_time_series.csv', index=False)

            break

if __name__ == '__main__':
    record_live()