

a=[]
f = open("align_lexicon.txt", "r")
for line in f.readlines():
	temp = line.split(' ')
	text=""
	for i in range(1,len(temp)-1):
		text=text+temp[i]+" "
	text=text+temp[len(temp)-1]
	a.append(text)
a.sort()
out=""
for i in range(0,len(a)):
	out=out+a[i]
print(out)
f.close()

