import typing
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(False)

class AudioReader:
    """Read audio file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """
    def __init__(
        self, 
        frame_length: int=256, 
        frame_step: int=160,
        fft_length: int=384,
        samplerate: int = 16000,
        *args, **kwargs
        ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.samplerate = samplerate

    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a WAV file
        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.
        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = librosa.load(wav_path) 

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram
    
    @staticmethod
    def get_mfcc(audio_path: str, samplerate: int = 16000, n_mfcc: int = 20) -> np.ndarray:
        """Get mfcc"""
        audio, orig_sr = librosa.load(audio_path)
        if orig_sr != samplerate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=samplerate)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc = n_mfcc)
        
        return mfcc

    @staticmethod
    def plot_raw_audio(wav_path: str, title:str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file
        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
        """
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = librosa.load(wav_path, sr=sr)

        duration = len(audio) / orig_sr

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title('Audio Plot')
        plt.ylabel('signal wave')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: typing.Union[np.ndarray, str], samplerate: int = None, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file
        Args:
            spectrogram (np.ndarray | str): Spectrogram of the audio file or path to file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        data = np.array([])
        if isinstance(spectrogram, str):
            data, sr = librosa.load(spectrogram)
            if samplerate is None:
                samplerate = sr
            if samplerate != sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=samplerate)
        else:
            data = spectrogram
            if samplerate is None:
                samplerate = 16000
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data, Fs=samplerate)
        plt.show()
        
    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.
        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.
        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return self.get_mfcc(audio_path, self.samplerate), label