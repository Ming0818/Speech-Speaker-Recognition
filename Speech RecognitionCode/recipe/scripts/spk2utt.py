

a={}
keys_list=[]
f = open("audio/dev.txt", "r")
for line in f.readlines():
    [ word, text] = line.split('/')
    [speaker, nohash , index_text] = text.split('_')
    [index,wav] = index_text.split('.')
    if speaker not in a.keys():
    	a[speaker]=[]

    a[speaker].append(speaker+'_'+word+'_'+index)
    	
keys_list=a.keys()
keys_list.sort()

for i in keys_list:
	out=i
	a[i].sort()
	for j in a[i]:
		out=out+' '+j
	print(out+' ')
f.close()

