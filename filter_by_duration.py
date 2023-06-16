import librosa

TRAIN = '/home/yehor/Work/github/vocos/tetiana-dataset/filelist.train'
VAL = '/home/yehor/Work/github/vocos/tetiana-dataset/filelist.val'

TRAIN_FIXED = '/home/yehor/Work/github/vocos/tetiana-dataset/filelist_fixed.train'
VAL_FIXED = '/home/yehor/Work/github/vocos/tetiana-dataset/filelist_fixed.val'

SR = 22050

MIN_DURATION = 1
MAX_DURATION = 15

with open(TRAIN) as f:
    filelist = f.read().splitlines()
    for audio_path in filelist:
        dur = librosa.get_duration(filename=audio_path, sr=SR)
        if dur < MAX_DURATION and dur > MIN_DURATION:
            with open(TRAIN_FIXED, 'a') as f:
                f.write(audio_path + '\n')


with open(VAL) as f:
    filelist = f.read().splitlines()
    for audio_path in filelist:
        dur = librosa.get_duration(filename=audio_path, sr=SR)
        if dur < MAX_DURATION and dur > MIN_DURATION:
            with open(VAL_FIXED, 'a') as f:
                f.write(audio_path + '\n')
