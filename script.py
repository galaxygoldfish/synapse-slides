from pyOpenBCI import cyton

def detect_blinks(sample):
    # Index of channel
    sample.channels_data[0]

board = cyton.Cyton()
board.start_streaming(detect_blinks)