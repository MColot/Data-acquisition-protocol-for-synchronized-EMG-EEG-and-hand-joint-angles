import mne


def sync(pathEEG):
    """
    cuts the EEG recording at the given path to match the length of the other recording (EMG and hand tracking)
    based on 2 triggers sent at the beginning and the end of the recording
    :param pathEEG: path to the file containing the EEG recording (.set file with corresponding .ftd file)
    """
    raw = mne.io.read_raw_eeglab(pathEEG, preload=True)
    events = mne.events_from_annotations(raw)
    triggerAnnotation = events[1]['123']
    triggerTimes = []
    for event in events[0]:
        if event[2] == triggerAnnotation:
            triggerTimes.append(event[0]/raw.info["sfreq"])

    # we only consider the first and the last triggers
    raw.crop(tmin=triggerTimes[0], tmax=triggerTimes[-1])
    raw.save(pathEEG[:-4] + "_sync_raw.fif")



if __name__ == "__main__":
    sync("S1_2_task2as int ICAout 6 20 23.set")