import librosa
import numpy
import librosa.display
from hmmlearn import hmm
import os

partitions = 15
features = 39
length = 200
X_input = []
Y_output = []
center_input_list = []
center_name_list = []
num_speakers = 40
num_neurons = 512


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
    for i in os.listdir(dir):
        s = dir + i
        a = mfcc_extractor(s)
        a = list(a)
        x = x + a
        lengths.append(len(a))
    y = numpy.array(x)
    return y,lengths

num = 0
for k in range(0,10000):
    if not (os.path.isdir('train/train'+str(k)+'/')):       
        continue
    if not (num < num_speakers):
        break
    (tr, trlengths) = return_allfeats('train/train'+str(k)+'/')
    b = []
    for i in range(0,len(tr)-partitions):
        x = tr[i]
        for j in range(1,partitions):
            x = numpy.concatenate([x,tr[i+j]])
        b.append(x)
    c = numpy.array(b)
    temp = []
    for i in range(0,int(len(c)/length)):
        X_input.append(c[(i)*length:(i+1)*length,:])
        temp.append(c[(i)*length:(i+1)*length,:])
        Y_output_temp = numpy.zeros(tuple([num_speakers]))
        Y_output_temp[num] = 1.0
        Y_output.append(Y_output_temp)
    center_input_list.append(numpy.array(temp))
    center_name_list.append(k)
    num = num + 1


import keras.layers

def splicer(x, **kwargs):
    index = kwargs.get("limit", None)
    x = x[ :, (index*features):((index+1)*features)]
    return x


def custompool(x):
    # import keras
    # x1 = x[ :, :, :512]
    # x2 = keras.backend.sqrt(keras.backend.abs(keras.backend.square(x[ :, :, :512]) - x[ :, :, 512:]))
    return x


Main_input = keras.layers.Input(shape=tuple([length, partitions*features]),  name='Main_input')


########################################################################################

############################
input_parser = keras.layers.Input(shape=tuple([partitions*features]), name='input_parser')

input_Processed_parser = keras.layers.Masking(mask_value=0)(input_parser)

parsed_output = [keras.layers.Lambda(splicer,output_shape=tuple([features]), arguments={"limit": x})(input_Processed_parser) for x in range(partitions)]

parser_model = keras.models.Model(inputs=input_parser, outputs=parsed_output, name="parser_model")
# ##############################

# ###########################
# # This returns a tensor
input_list = []
for i in range(0, 15):
    input_list.append(keras.layers.Input(shape=tuple([features])))

hidden_layer1_list = []

for i in range(2, 13):
    hidden_layer1_list.append(keras.layers.Dense(num_neurons, activation='tanh')(keras.layers.concatenate([input_list[i-2], input_list[i-1], input_list[i], input_list[i+1], input_list[i+2]])))

hidden_layer2_list = list()
hidden_layer2_list.append(keras.layers.Dense(num_neurons, activation='tanh')(keras.layers.concatenate([hidden_layer1_list[0], hidden_layer1_list[2], hidden_layer1_list[4]])))
hidden_layer2_list.append(keras.layers.Dense(num_neurons, activation='tanh')(keras.layers.concatenate([hidden_layer1_list[3], hidden_layer1_list[5], hidden_layer1_list[7]])))
hidden_layer2_list.append(keras.layers.Dense(num_neurons, activation='tanh')(keras.layers.concatenate([hidden_layer1_list[6], hidden_layer1_list[8], hidden_layer1_list[10]])))


hidden_layer3 = keras.layers.Dense(num_neurons, activation='tanh')(keras.layers.concatenate([hidden_layer2_list[0], hidden_layer2_list[1], hidden_layer2_list[2]]))
hidden_layer4 = keras.layers.Dense(num_neurons, activation='tanh')(hidden_layer3)
hidden_layer5 = keras.layers.Dense(num_neurons, activation='tanh')(hidden_layer4)

# the Input layer and three Dense layers
voice_model = keras.models.Model(inputs=input_list, outputs=hidden_layer3)

final_model = keras.models.Model(inputs=parser_model.input, outputs=voice_model(parser_model.output))
########################3
#################################################################################


	
TimeDistributed_layer = keras.layers.TimeDistributed(final_model)(Main_input)

Mean = keras.layers.AveragePooling1D(pool_size=tuple([length]), strides=None)(TimeDistributed_layer)

squareSum = keras.layers.AveragePooling1D(pool_size=tuple([length]), strides=None)(keras.layers.Lambda(lambda x: keras.backend.square(x))(TimeDistributed_layer))



Pooling1D = keras.layers.Lambda(custompool)(keras.layers.concatenate([Mean, squareSum]))

Embedding_A = keras.layers.Dense(num_neurons, activation='tanh', name='Embedding_A')(keras.layers.Flatten()(Pooling1D))

Embedding_B = keras.layers.Dense(num_neurons, activation='tanh', name='Embedding_B')(Embedding_A)

out = keras.layers.Dense(num_speakers, activation='softmax')(Embedding_B)

Final_model = keras.models.Model(inputs=[Main_input], outputs=[out])
EmbeddingA_model = keras.models.Model(inputs=[Main_input], outputs=[Embedding_A])
EmbeddingB_model = keras.models.Model(inputs=[Main_input], outputs=[Embedding_B])


sgd = keras.optimizers.SGD(lr=1, decay=0.0)
Final_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

X_input = numpy.array(X_input)
Y_output = numpy.array(Y_output)

s = numpy.arange(X_input.shape[0])

X_input = X_input[s]
Y_output = Y_output[s]
Final_model.fit(x=X_input,y=Y_output, epochs=10)

center_list = []
center_radius_list = []
for i in range(0,len(center_input_list)):
    center_outputs = EmbeddingA_model.predict(center_input_list[i])
    temp_center_outputs = numpy.mean(center_outputs, axis=0)
    center_list.append(temp_center_outputs)
    # change radius for better results
    center_radius_list.append(numpy.max(numpy.sum((center_outputs - temp_center_outputs)**2,axis=1)))

Pred_list = []
Actual_List = []

# not well separated model is a failure

for i in os.listdir("test"):
    if int(i.split("-")[0]) in center_name_list:
        test_feature = mfcc_extractor("test/"+i)
        # print(test_feature.shape)
        if test_feature.shape[0] < length :
            continue
        b = []
        for k in range(0,len(test_feature)-partitions):
            x = test_feature[k]
            for j in range(1,partitions):
                x = numpy.concatenate([x,test_feature[k+j]])
            b.append(x)
        b = numpy.array(b)
        if b.shape[0] < length :
            continue
        # print(b.shape)
        val = EmbeddingA_model.predict(numpy.array([b[:length,:]]))
        for j in range(0,len(center_list)):
            k = numpy.sum((center_list[j] - val)**2,axis=1)
            if k < center_radius_list[j]:
                print(i)
                Pred_list.append(center_name_list[j])
                Actual_List.append(int(i.split("-")[0]))
                print(center_name_list[j],int(i.split("-")[0]))

if len(Actual_List) != len(Pred_list) :
    print("error")

total = len(Actual_List)
count = 0.0
for i in range(0,total):
    if Actual_List[i] == Pred_list[i] :
        count = count+1.0

print((count/total)*100)