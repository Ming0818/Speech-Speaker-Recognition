import os
for root, directories, filenames in os.walk('LibriSpeech/'):
	# for directory in directories:
		# print(os.path.join(root, directory)) 
	for filename in filenames:
		if ".txt" not in filename and ".TXT" not in filename:
			# print(filename.split("-")[0])
			os.system("mkdir -p train/train"+filename.split("-")[0])
			a = str(os.path.join(root,filename))
			os.system("cp " + a + " " + "train/train"+filename.split("-")[0] )