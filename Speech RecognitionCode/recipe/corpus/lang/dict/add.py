ns = {}
with open('nonsilence_phones.txt', 'r') as f:
	x = f.readlines()
	for y in x:
		y = y.split('\n')
		ns[y[0]]=1
	f.close()
os = {}
with open('optional_silence.txt', 'r') as f:
	x = f.readlines()
	for y in x:
		y = y.split('\n')
		os[y[0]]=1
	f.close()
sp = {}
with open('silence_phones.txt', 'r') as f:
	x = f.readlines()
	for y in x:
		y = y.split('\n')[0]
		sp[y]=1
	f.close()
s = {}
with open('lexicon.txt', 'r') as f:
	x = f.readlines()
	for y in x:
		y = y.split(' ')
		for z in range(1, len(y)-1):
			if y[z] not in ns.keys() and y[z] not in os.keys() and y[z] not in sp.keys():
				s[y[z]] = 1
	f.close()
with open('nonsilence_phones.txt', 'w') as f:
	for a in ns.keys():
		f.write(a+"\n")
	for a in s.keys():
		f.write(a+"\n")
	f.close()