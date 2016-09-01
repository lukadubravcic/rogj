import sys
import os
import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scikits.audiolab import wavread

from funkcije import *




print 'Uzorak govora:', str(sys.argv[1])
print 'Lab transkript:', str(sys.argv[2])


data, sample_rate, encoding = wavread(sys.argv[1])

lab_file = open(sys.argv[2], "r")

emph_signal = signal_emphasis(data)

#racunanje mfc koeficijenata nad cijelim zapisom


frames = framing(emph_signal, sample_rate)

apply_window(frames)


NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  


filter_bank = filter_banks(sample_rate, pow_frames)

num_ceps = 12
mfcc = dct(filter_bank, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] 

#primjer ispisa mfcc-a za prvih 5 okvira (okvir = 25ms)
print mfcc[:5]