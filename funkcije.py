import sys
import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def signal_emphasis(signal, pre_emphasis = 0.97):	
	return numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


def framing(emphasized_signal, sample_rate, frame_size = 0.025, frame_stride = 0.01):

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
	signal_length = len(emphasized_signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  
	pad_signal_length = num_frames * frame_step + frame_length
	z = numpy.zeros((pad_signal_length - signal_length))
	pad_signal = numpy.append(emphasized_signal, z) 

	indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(numpy.int32, copy=False)]

	return frames


def apply_window(frames, frame_length = 400):
	frames *= numpy.hamming(frame_length)
	return frames
	

def filter_banks(sample_rate, pow_frames, nfilt = 40, NFFT = 512):
	low_freq_mel = 0
	high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  
	mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
	hz_points = (700 * (10**(mel_points / 2595) - 1))  
	bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

	fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
	    f_m_minus = int(bin[m - 1])  
	    f_m = int(bin[m])             
	    f_m_plus = int(bin[m + 1])    

	    for k in range(f_m_minus, f_m):
	        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	    for k in range(f_m, f_m_plus):
	        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

	filter_banks = numpy.dot(pow_frames, fbank.T)
	filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  
	filter_banks = 20 * numpy.log10(filter_banks)  

	return filter_banks
