import os
root = "train"
for i in os.listdir(root):
	i = str(os.path.join(root,i))
	os.system("mkdir -p test")
	w = os.listdir(i)
	jj = 0
	for q in w:
		if(jj > (len(w)/10.0)):
			break
		jj = jj+1
		a = i + "/"+q
		os.system("mv "+ a + " test")

