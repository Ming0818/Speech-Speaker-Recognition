

a=[]
f = open("audio/dev.txt", "r")
for line in f.readlines():
    [ word, text] = line.split('/')
    [speaker, nohash , index_text] = text.split('_')
    [index,wav] = index_text.split('.')
    a.append(speaker+'_'+word+'_'+index+' '+'corpus/data/wav/'+word+'/'+speaker+'_nohash_'+index+'.wav')
a.sort()
for i in range(0,len(a)):
	print(a[i])
f.close()