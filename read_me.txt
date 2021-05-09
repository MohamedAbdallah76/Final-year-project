Data Augmentation:
	Parameters: (alpha_sr, alpha_rd, alpha_ri, num_aug)
	the data augmentation can be run through the aug_main.py file. To run load the intended file 
	through line 8 where the pd.read_csv can be found then simply run the file.

Feature Extraction and preprocessing:
	 the feature ectraction can be rum through the feature_extraction_main.py file.
	 to run load the appropriat files through the pd.read_csv in line 11 and 13.
	 when producing the training file after loading the csv file simpy run and a csv file with the features will be produced.
	 when producing the test file COmment out lines 17 and 22 and in line 48 replace data_array with data_array_v then in lines 70 and 72 remove any label related data.

Artificial neural network:
	To run open the ANN.py file then  input the directory of the inteded files in lines 27, 29, and 31 where the pd.read_csv can be found then run the file and observe the results.