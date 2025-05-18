import pandas as pd
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' if running in a non-GUI environment (like a server)
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pyautogui
import os # For saving plots in a local pdf files
from utils import update_buffer, get_last_data
from pylsl import StreamInlet, resolve_byprop

# app = Flask(__name__) # Not used in this version

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
BLINK_MIN_THRESHOLD = 250
BLINK_MAX_THRESHOLD = 1200
SEQUENCE_COUNT = 2
MIN_BLINK_INTERVAL = 4
INDEX_CHANNEL = [1]
RUNTIME_SECONDS = 60 # Reduced for quicker testing

# Define plot directory (useful if you save plots)
plot_dir = 'plots_output' # Name of the directory to save plots
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# --- Define Theme Colors (from your CSS :root for consistency) ---
THEME_COLORS = {
    'primary_glow': '#00ffff',   # Bright Cyan
    'secondary_glow': '#ff00ff', # Bright Magenta
    'background_body': '#0a0f1f', # Darker, slightly blueish background
    'background_card': '#141a2e', # Card background
    'text_primary': '#e0e0e0',    # Light grey text
    'text_muted': '#a0a0c0',      # Muted text / axis labels
    'axis_edge': 'rgba(0, 255, 255, 0.4)', # Primary glow for axes lines (slightly more visible)
    'grid_lines': 'rgba(0, 255, 255, 0.15)' # Subtle grid lines
}

def apply_futuristic_theme_to_plot():
    """Applies a consistent futuristic theme to Matplotlib's rcParams."""
    plt.style.use('seaborn-v0_8-darkgrid') # Start with a dark base
    plt.rcParams.update({
        'figure.facecolor': THEME_COLORS['background_body'],
        'axes.facecolor': THEME_COLORS['background_card'],
        'text.color': THEME_COLORS['text_primary'],
        'axes.labelcolor': THEME_COLORS['text_muted'],
        'xtick.color': THEME_COLORS['text_muted'],
        'ytick.color': THEME_COLORS['text_muted'],
        'axes.edgecolor': THEME_COLORS['axis_edge'],
        'grid.color': THEME_COLORS['grid_lines'],
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.titlecolor': THEME_COLORS['primary_glow'], # For main titles on axes
        'axes.titlesize': 'large',
        'axes.labelsize': 'medium',
        'legend.frameon': False, # No frame around legend
        'legend.labelcolor': THEME_COLORS['text_muted'], # Legend text color
        'legend.facecolor': THEME_COLORS['background_card'], # Legend background
    })


def bandpass(data, fs, low=0.1, high=5.5, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=0)

def connect_eeg():
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    else:
        print('Found it!')
    inlet = StreamInlet(streams[0], max_chunklen=12)
    print("Start acquiring data")
    return inlet

def record_live():
    inlet = connect_eeg()
    info = inlet.info()
    fs = int(info.nominal_srate())
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None
    last_blink_time = 0
    blink_count = 0
    start = time.time()
    whole_data = np.array([])
    whole_timestamps = np.array([]) 
    blink_data_all = np.array([])
    blink_time_all = np.array([])
    blink_times_sequence = [] # Renamed from blink_times for clarity

    print(f"Starting recording for {RUNTIME_SECONDS} seconds...")

    while True:
        now_time = time.time()
        eeg_data, timestamp_chunk = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        if len(eeg_data) == 0:
            continue

        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
        
        eeg_buffer, filter_state = update_buffer(
            eeg_buffer, ch_data, notch=True, filter_state=filter_state
        )
        
        data_epoch = get_last_data(eeg_buffer, EPOCH_LENGTH * fs)

        # Generate timestamps for the current epoch
        # If timestamp_chunk from pull_chunk is reliable and matches data_epoch length, use it.
        # Otherwise, generate based on now_time and fs.
        # For simplicity, generating based on now_time and known epoch length.
        # This needs to be accurate for time-series plots.
        # The original code had a potential issue if len(data_epoch) didn't match SHIFT_LENGTH*fs exactly.
        current_epoch_timestamps = np.linspace(now_time - EPOCH_LENGTH, now_time, len(data_epoch))

        filtered_epoch = bandpass(data_epoch, fs)
        filtered_epoch_abs = np.abs(filtered_epoch)

        if np.max(filtered_epoch_abs) > BLINK_MIN_THRESHOLD and \
        np.max(filtered_epoch_abs) < BLINK_MAX_THRESHOLD:
            if now_time - last_blink_time > MIN_BLINK_INTERVAL:
                print(f"Blink detected! ({blink_count + 1})")
                blink_count += 1
                last_blink_time = now_time
                blink_times_sequence.append(now_time)
                
                # Sequence detection window logic (from previous Flask example)
                sequence_detection_window = MIN_BLINK_INTERVAL * (SEQUENCE_COUNT + 0.5)
                blink_times_sequence = [t for t in blink_times_sequence if now_time - t <= sequence_detection_window]

                if len(blink_times_sequence) >= SEQUENCE_COUNT:
                    sequence_blink_action()
                    blink_times_sequence.clear()
                else:
                    single_blink_action()
        
        # Data accumulation
        flat_filtered_epoch = filtered_epoch.flatten() # Ensure 1D for appending
        if len(whole_data) == 0:
            whole_data = flat_filtered_epoch
            whole_timestamps = current_epoch_timestamps
        else:
            whole_data = np.append(whole_data, flat_filtered_epoch)
            whole_timestamps = np.append(whole_timestamps, current_epoch_timestamps)

        # Blink data accumulation
        blink_mask = (np.abs(flat_filtered_epoch) > BLINK_MIN_THRESHOLD) & \
                     (np.abs(flat_filtered_epoch) < BLINK_MAX_THRESHOLD)
        if blink_mask.any():
            blink_data_all = np.append(blink_data_all, flat_filtered_epoch[blink_mask])
            blink_time_all = np.append(blink_time_all, current_epoch_timestamps[blink_mask])

        if time.time() > start + RUNTIME_SECONDS:
            print("Runtime reached. Generating plots and exiting.")
            if len(whole_data) > 0:
                # Pass fs to time_to_frequency_domain if it needs it
                freqs, normalized_mag, mag = time_to_frequency_domain(whole_data, fs) # Added fs
                
                plot_entire_data(freqs, normalized_mag, whole_data, whole_timestamps, now_time, mag)
                plot_blink_selected_data(blink_data_all, fs, freqs, normalized_mag, whole_timestamps, whole_data, now_time, blink_time_all)
                export_raw_csv(blink_time_all, blink_data_all, now_time) # Added now_time
            else:
                print("No data collected to plot.")
            break
    print(f"Total blinks recorded: {blink_count}")


def plot_entire_data(freqs, normalized_mag, whole_data, whole_timestamps, now_time, mag):
    apply_futuristic_theme_to_plot() # Apply theme

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False) # Use fig, ax for more control
    fig.suptitle("Full Session EEG Analysis", fontsize=18, color=THEME_COLORS['primary_glow'], fontweight='bold', fontfamily='Orbitron')

    # --- First subplot (Frequency Domain) ---
    ax1.plot(freqs, normalized_mag, color=THEME_COLORS['primary_glow'], linewidth=1.5)
    ax1.set_title("Normalized Frequency Spectrum (All Data)", color=THEME_COLORS['primary_glow'])
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_xlim([0, 15]) # Extended slightly for better view
    ax1.set_ylabel("Normalized Magnitude")
    ax1.grid(True, linestyle=':', alpha=0.5, color=THEME_COLORS['grid_lines'])


    # --- Second subplot (Time Domain) ---
    ax2.plot(whole_timestamps, whole_data, color=THEME_COLORS['primary_glow'], linewidth=1)
    ax2.set_title("Raw EEG Time Series (Filtered)", color=THEME_COLORS['primary_glow'])
    ax2.set_xlabel("Time (s, relative to start of epoch chunks)")
    ax2.set_ylabel("Amplitude (µV)")
    ax2.grid(True, linestyle=':', alpha=0.5, color=THEME_COLORS['grid_lines'])

    # Adjust layout to prevent overlap and accommodate suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect=[left, bottom, right, top]

    filename = f"all_data_plot_{int(now_time)}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), dpi=150) # Use fig facecolor
    plt.close(fig) # Close the figure to free memory
    print(f"Saved plot: {filepath}")

    # Optional: Save CSV data
    # df_freq = pd.DataFrame({'frequencies': freqs, 'magnitude': mag})
    # df_freq.to_csv(os.path.join(plot_dir, f'freq_data_{int(now_time)}.csv'), index=False)
    # df_time = pd.DataFrame({'timestamp': whole_timestamps, 'eeg_value': whole_data})
    # df_time.to_csv(os.path.join(plot_dir,f'time_data_{int(now_time)}.csv'), index=False)


def plot_blink_selected_data(blink_data_all, fs, freqs_all, normalized_mag_all, whole_timestamps, whole_data, now_time, blink_time_all):
    apply_futuristic_theme_to_plot() # Apply theme

    n_blink = len(blink_data_all)
    if n_blink > 1: # Need more than 1 sample for FFT
        fft_blink  = np.fft.rfft(blink_data_all)
        mag_blink  = np.abs(fft_blink) / n_blink
        freqs_blink = np.fft.rfftfreq(n_blink, d=1/fs)
        mag_blink_max = np.max(mag_blink) if np.max(mag_blink) > 0 else 1
        normalized_mag_blink = mag_blink / mag_blink_max
    else:
        freqs_blink, normalized_mag_blink = np.array([0]), np.array([0])


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Blink Event Analysis", fontsize=18, color=THEME_COLORS['primary_glow'], fontweight='bold', fontfamily='Orbitron')

    # --- Subplot 1: All Samples Frequency Domain ---
    ax1.plot(freqs_all, normalized_mag_all, color=THEME_COLORS['primary_glow'], linewidth=1.5)
    ax1.set_title('Full Spectrum (All Samples)', color=THEME_COLORS['primary_glow'])
    ax1.set_ylabel('Norm. Magnitude')
    ax1.set_xlim(0, 15)
    ax1.grid(True, linestyle=':', alpha=0.5, color=THEME_COLORS['grid_lines'])

    # --- Subplot 2: BLINK-ONLY Samples Frequency Domain ---
    ax2.plot(freqs_blink, normalized_mag_blink, color=THEME_COLORS['secondary_glow'], linewidth=1.5) # Magenta for blinks
    ax2.set_title('Blink Event Spectrum', color=THEME_COLORS['secondary_glow'])
    ax2.set_ylabel('Norm. Magnitude')
    ax2.set_xlabel('Frequency (Hz)') # Add x-label here if not sharing x-axis
    ax2.set_xlim(0, 15)
    ax2.grid(True, linestyle=':', alpha=0.5, color=THEME_COLORS['grid_lines'])

    # --- Subplot 3: Time-Series with Blink Samples Marked ---
    ax3.plot(whole_timestamps, whole_data, label='Full EEG Stream', color=THEME_COLORS['primary_glow'], alpha=0.7, linewidth=1)
    if len(blink_time_all) > 0 and len(blink_data_all) > 0:
        ax3.scatter(blink_time_all, blink_data_all, color=THEME_COLORS['secondary_glow'], s=50, label='Blink Samples', zorder=5, edgecolors=THEME_COLORS['background_card'], linewidths=0.5)
    ax3.set_title('Time Series with Blinks Highlighted', color=THEME_COLORS['primary_glow'])
    ax3.set_xlabel('Time (s, relative to start of epoch chunks)')
    ax3.set_ylabel('Amplitude (µV)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.5, color=THEME_COLORS['grid_lines'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"blink_vs_all_plot_{int(now_time)}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"Saved plot: {filepath}")
    # plt.show() # Remove or comment out if running headlessly or with Agg backend


def export_raw_csv(blink_time_all, blink_data_all, now_time):
    if len(blink_time_all) > 0 and len(blink_data_all) > 0:
        df = pd.DataFrame({'t': blink_time_all, 'eeg': blink_data_all})
        csv_filename = f'blink_only_time_series_{int(now_time)}.csv'
        filepath = os.path.join(plot_dir, csv_filename)
        df.to_csv(filepath, index=False)
        print(f"Exported CSV: {filepath}")


def time_to_frequency_domain(whole_data, fs): # Added fs argument
    n = len(whole_data)
    if n == 0:
        return np.array([0]), np.array([0]), np.array([0])
    fft_vals = np.fft.rfft(whole_data)
    mag = np.abs(fft_vals) / n
    # freqs = np.fft.rfftfreq(n, d=1/260) # Original used fixed 260Hz
    freqs = np.fft.rfftfreq(n, d=1/fs) # Use actual sampling rate
    mag_max = np.max(mag) if np.max(mag) > 0 else 1
    normalized_mag = mag / mag_max
    # Return freqs first, then normalized_mag, then mag (consistent order)
    return freqs, normalized_mag, mag


def single_blink_action():
    print(">>> ACTION: Single blink - Right Arrow")
    pyautogui.press("right")

def sequence_blink_action():
    print(">>> ACTION: Double blink - Play/Pause Media")
    pyautogui.press("mediaplaypause")

if __name__ == '__main__':
    try:
        record_live()
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        print("Program terminated.")