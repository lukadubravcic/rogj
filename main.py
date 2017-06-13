
import functions
import os, sys
import numpy as np
import scipy.spatial 


classes = ['a', 'a:', 'b', 'c', 'cc', 'C', 'd', 'dz', 'DZ', 'e', 'e:', 'f', 'g', 'h', 'i', 'i:', 'j',
	       'k', 'l', 'L', 'm', 'n' ,'N', 'o', 'o:', 'p', 'r', 'r:', 's', 'S', 't', 'u', 'u:', 'v', 'z', 'Z']
mean_vectors = []
cov_mats = []

		
def learning(directory):
	
	global mean_vectors
	global cov_mats
	
	mfcc_data = []
	phonems = []
	
	files = wav_files(directory)	
	
	for file in files:
		filename = directory + file[:-4]
		data, chars = functions.getMfcc(filename)

		mfcc_data.extend(data)
		phonems.extend(chars)
		
	mean_vectors, cov_mats = functions.getMeanCov(mfcc_data, phonems)
		
	return 1


def classification(directory):
	
	files = wav_files(directory)
	char_counter = 0
	hit_counter = 0

	for file in files:
		
		filename = file[:-4]
		mfcc, phonems = functions.getMfcc(directory + filename)		
		m_vector = functions.getMeanVectors(mfcc, phonems)	

		for x in xrange(0,len(phonems)):

			char_counter += 1
			distance = scipy.spatial.distance.mahalanobis(mean_vectors[0], m_vector[0], np.linalg.inv(cov_mats[0]))						

			for y in xrange(0, len(classes)):

				if (len(mean_vectors[y]) > 1):					
					tmp = scipy.spatial.distance.mahalanobis(mean_vectors[y], m_vector[x], np.linalg.inv(cov_mats[y]))					
					if tmp < distance:
						distance = tmp
						index = y
			
			if classes[index] == phonems[x]:
				hit_counter += 1

	s_rate = hit_counter * 100 / char_counter
	print "Correctly classified:", hit_counter, "/", char_counter, "(", s_rate, "%)"
	return 1	


def wav_files(directory):
	files = [file for file in os.listdir(directory) if file.endswith('.wav')]
	return files;

        	
def main(argv):
	
	test_dir = argv[0]
	train_dir = argv[1]	
	
	learning(train_dir)
	classification(test_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
