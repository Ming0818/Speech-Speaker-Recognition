import librosa
import numpy
import librosa.display
from hmmlearn import hmm
import os
import warnings
warnings.filterwarnings("ignore")

# number of states per HMM
M = 50

#number of speakers to consider
Num_Speakers = 40

#number of utts to take per speaker
num_utts = 10


def mfcc_extractor(a):
    y, sr = librosa.load(a,sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13,hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    mfccdelta = librosa.feature.delta(mfcc)
    mfccdeltadelta = librosa.feature.delta(mfcc, order=2)
    final = numpy.append(mfcc, mfccdelta,axis=0)
    final = numpy.append(final, mfccdeltadelta,axis=0)
    return final.T

def return_allfeats(dir):
    lengths = []
    x = []
    j = 0
    for i in os.listdir(dir):
        if j > num_utts:
            break
        j = j + 1
        s = dir + i
        a = mfcc_extractor(s)
        a = list(a)
        x = x + a
        lengths.append(len(a))
    y = numpy.array(x)
    return y,lengths


def create_model(M,i):
    (tr, trlengths) = return_allfeats('train/train'+str(i)+'/');
    model = hmm.GaussianHMM(n_components=M, covariance_type="diag", init_params='mcs', params='mcs', n_iter=10,
                             tol=1e-7, verbose=True)
    model.transmat_ = numpy.ones((M, M), dtype='float') / M
    model.fit(tr, trlengths)
    return model

numpy.random.seed(42)
Model_list = []
speakers = []
num = 0 
for i in range(0,10000):
    if not (os.path.isdir('train/train'+str(i)+'/')):       
        continue
    if not (num < Num_Speakers):
        break
    print(i)
    speakers.append(i)
    num = num + 1
    Model_list.append([create_model(M,i),i])

Pred_list = []
Actual_List = []

for i in os.listdir("test"):
    if int(i.split("-")[0]) in speakers:
        Maximum = -float('inf')
        Max_index = 0
        for j in range(0,len(Model_list)):
            k = Model_list[j][0].score(mfcc_extractor("test/"+i)[:200,:])
            if Maximum < k:
                Max_index = j
                Maximum = k
        print(i)
        Pred_list.append(Model_list[Max_index][1])
        Actual_List.append(int(i.split("-")[0]))
        print(Model_list[Max_index][1],i)

print(Actual_List)
print(Pred_list)

if len(Actual_List) != len(Pred_list) :
    print("error")

total = len(Actual_List)
count = 0.0
for i in range(0,total):
    if Actual_List[i] == Pred_list[i] :
        count = count+1.0

print((count/total)*100)