import sys, os
import numpy 
from scipy.io import wavfile
from scipy.fftpack import dct


classes = ['a', 'a:', 'b', 'c', 'cc', 'C', 'd', 'dz', 'DZ', 'e', 'e:', 'f', 'g', 'h', 'i', 'i:', 'j',
	       'k', 'l', 'L', 'm', 'n' ,'N', 'o', 'o:', 'p', 'r', 'r:', 's', 'S', 't', 'u', 'u:', 'v', 'z', 'Z'  ]


def signal_emphasis(signal, pre_emphasis = 0.97):	
	return numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


def framing(emphasized_signal, sample_rate, frame_size = 0.025, frame_stride = 0.01):

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # sekunde -> uzorke
	signal_length = len(emphasized_signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  

	pad_signal_length = num_frames * frame_step + frame_length
	z = numpy.zeros((pad_signal_length - signal_length))
	pad_signal = numpy.append(emphasized_signal, z) # nadopuni nulama okvir koji je kraci od zadane duljine

	indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(numpy.int32, copy=False)]
	
	return frames


def apply_window(frames, frame_length = 400):
	frames *= numpy.hamming(frame_length)
	
	return frames
	

def filter_banks(sample_rate, pow_frames, nfilt = 26, NFFT = 512):
	low_freq_mel = 0
	high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Hz -> Mel
	mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # jednaka udaljenost
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Mel -> Hz
	bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

	fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
	    f_m_minus = int(bin[m - 1])   # lijevo
	    f_m = int(bin[m])             # centar
	    f_m_plus = int(bin[m + 1])    # desno

	    for k in range(f_m_minus, f_m):
	        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	    for k in range(f_m, f_m_plus):
	        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

	filter_banks = numpy.dot(pow_frames, fbank.T)
	filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)
	filter_banks = 20 * numpy.log10(filter_banks)  # dB

	return filter_banks


def getMfcc(filename):

	chars = []
	mfcc_data = []
	sample_rate, wav_file = wavfile.read(filename + ".wav")
	lab_file = open(filename + ".lab")
	
	for ln in lab_file.readlines():

		start, end, char = ln.split()
		start = int(start) * sample_rate / 10000000
		end = int(end) * sample_rate / 10000000 - 1		

		phonem_sig = numpy.float64(wav_file[start:end])				
		emph_signal = signal_emphasis(phonem_sig)
		frames = framing(emph_signal, sample_rate)
		apply_window(frames)

		NFFT = 512
		mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # fft magnituda
		pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # spektar snage

		filter_bank = filter_banks(sample_rate, pow_frames)

		num_ceps = 12
		mfcc = dct(filter_bank, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # 2-13 koef. zadrzavamo
		
		cep_lifter = 22
		(nframes, ncoeff) = mfcc.shape
		n = numpy.arange(ncoeff)
		lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
		mfcc *= lift 		
		
		if char in classes:			
			chars.append(char)
			mfcc_data.append(mfcc)
			
	return mfcc_data, chars


def getMeanCov(mfcc_data, phonems):

	mean_vectors = []
	cov_mats = []	
					
	for x in xrange(0, len(classes)):
		
		vector = []
		tmp2 = []
		tmp = []

		for y in xrange(0,len(phonems)):
						
			if(phonems[y] == classes[x]):
				tmp.extend(mfcc_data[y])		

		if ((len(tmp))!=0): # ako slovo postoji u obradenim fonemima
			tmp2 = numpy.asmatrix(tmp)		
			
			for g in xrange(0,12): #vektor srednjih vrijednosti
				suma = sum(tmp2[:,g]) / len(tmp2)
				vector.append(suma[0, 0])			 
			
			mean_vectors.append(vector)
			cov_mats.append(numpy.cov(tmp2, rowvar=False)) #racunanje kovarijance
			
		else:
			print "Phonem '", classes[x] , "' doesnt exist in training set and wont be classified."
			mean_vectors.append([0])
			cov_mats.append([0])

	return mean_vectors, cov_mats


def getMeanVectors(mfcc_data, phonems):

	tmp = []
	m_vector = []
	vector_list = []
	
	for x in xrange(0,len(phonems)):
		
		m_vector = []
		tmp = []
		tmp.extend(mfcc_data[x])
		tmp = numpy.asmatrix(tmp)

		for y in xrange(0,12):

			suma = sum(tmp[:,y]) / len(tmp)			
			m_vector.append(suma[0, 0])
		
		vector_list.append(m_vector)
		
	return vector_list
						
		