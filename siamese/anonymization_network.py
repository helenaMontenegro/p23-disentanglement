#########################
## Load trained models ##
#########################

import os, random, h5py, numpy as np, math, matplotlib, tensorflow as tf, shutil
from skimage.metrics import structural_similarity
from skimage.filters import gaussian
os.environ["KERAS_BACKEND"] = "tensorflow"
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Lambda, Input, Reshape, Dense, Dropout, Activation, concatenate, Add
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.losses import cosine_distance
from tensorflow.nn import l2_normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## variables ##
batch_size = 32                         # batch size
epochs = 1000                           # epochs
grayscale = False                       # auxiliary variable that differentiates between grayscale and rgb
opt = Adam(lr=1e-4, decay=1e-6)         # optimizer
latent_dim = 512                        # dimensions of latent vectors
identity_latent_dim = 64                # dimensions of latent space of anonymization VAE
pre_trained_model_ee = 20               # name of pretrained disentanglement model weights
path_models = 'models'                  # folder that contains pretrained model weights

#############
# Load data #
#############
f = h5py.File('celeba_hq_big.hdf5')

label1 = f['ids']                 # labels for the identity recognition network
label1 = np.asarray(label1)
label2 = f['atts']                # labels for the task-related classification network
label2 = np.asarray(label2)[:, 31]
label3 = f['set']                # labels for the dataset to which the sample belongs (train - 0, test - 1 or validation - 2)
label3 = np.asarray(label3)
x = f['images']                  # image data
x = np.asarray(x)
f.close()
if len(x.shape) == 3:
    grayscale = True
    x = np.reshape(x, (-1, 64, 64, 1))

# normalize image data
x = (x - 127.5) / 127.5
x = x.astype('float16')

# split data in train, test, validation sets
idx_train = np.asarray(np.where(label3 == 0)[0])
idx_valid  = np.asarray(np.where(label3 == 1)[0])
idx_test  = np.asarray(np.where(label3 == 2)[0])

x_train = x[idx_train]
x_test = x[idx_test]
x_valid = x[idx_valid]
y_train1 = label1[idx_train]
y_test1 = label1[idx_test]
y_valid1 = label1[idx_valid]
y_train2 = label2[idx_train]
y_test2 = label2[idx_test]
y_valid2 = label2[idx_valid]

input_shape = x_train.shape[1:]
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

##########################
# Load pretrained models #
##########################

def res_block(x, filters_out, kernel_size=(3, 3), resample=None):
    if resample == None:
        shortcut = x
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'downsample':
        shortcut = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        shortcut = BatchNormalization()(shortcut)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x) 
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    elif resample == 'upsample':
        shortcut = UpSampling2D()(x)
        shortcut = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = UpSampling2D()(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters_out, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.2)(x)
    return x   

def model_feature_extractor():
    d0 = Input((input_shape))

    h = Conv2D(latent_dim // 8, (3, 3), strides=(2, 2), padding='same')(d0)
    h = LeakyReLU(0.2)(h)
    
    h = res_block(h, latent_dim // 4, resample='downsample')
    h = res_block(h, latent_dim // 4)

    h = res_block(h, latent_dim // 2, resample='downsample')
    h = res_block(h, latent_dim // 2)
    
    h = res_block(h, latent_dim, resample='downsample')
    h = res_block(h, latent_dim)
    
    h = res_block(h, latent_dim * 2, resample='downsample')
    h = res_block(h, latent_dim * 2)

    h1 = GlobalAveragePooling2D()(h)
    medical_features = Dense(latent_dim, name="medical_features")(h1)
    identity_features = Dense(latent_dim, name="identity_features")(h1)
    other_features = Dense(latent_dim, name="other_features")(h1)

    feature_extractor = Model(d0, [medical_features, identity_features, other_features])
    feature_extractor.compile(loss='binary_crossentropy', optimizer=opt)

    return feature_extractor

def model_disease_classifier():
    input_vector = Input((latent_dim,))
    h = Dense(latent_dim // 2)(input_vector)
    h = Dropout(0.5)(h)
    h = Dense(1, activation='sigmoid')(h)
    classifier = Model(input_vector, h, name='dis_classifier')
    classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return classifier

def model_generator():
    units = 256
    medical_features = Input(shape=(latent_dim,))
    identity_features = Input(shape=(latent_dim,))
    other_features = Input(shape=(latent_dim,))
    input_vector = concatenate([medical_features, identity_features, other_features])
    h = Dense(input_shape[0]//16 * input_shape[1]//16 * units)(input_vector)
    h = LeakyReLU(0.2)(h)
    h = Reshape((input_shape[0]//16, input_shape[1]//16, units))(h)
    
    h = res_block(h, units // 2, resample='upsample')
    h = res_block(h, units // 2)

    h = res_block(h, units // 4, resample='upsample')
    h = res_block(h, units // 4)
    
    h = res_block(h, units // 8, resample='upsample')
    h = res_block(h, units // 8)
    
    h = res_block(h, units // 16, resample='upsample')
    h = res_block(h, units // 16)
    
    depth = 3
    if grayscale:
        depth = 1

    h = Conv2D(depth, (3, 3), strides=(1, 1), padding='same', activation='tanh')(h)  # 8*6*64
    
    generator = Model([medical_features, identity_features, other_features], h, name="Generator")
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    return generator

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], identity_latent_dim), mean=0., stddev=1)
    return z_mean + K.exp(z_log_sigma / 2) * epsilon

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

feature_extractor = model_feature_extractor()
generator = model_generator()
disease_classifier = model_disease_classifier()

def load_pretrained_models(ee):
    feature_extractor.load_weights('./' + path_models + '/feature_extractor_weights_'+ str(ee) +'.h5')
    make_trainable(feature_extractor, False)
    feature_extractor.compile(loss='binary_crossentropy', optimizer=opt)
    generator.load_weights('./' + path_models + '/generator_weights_'+ str(ee) +'.h5')
    make_trainable(generator, False)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    disease_classifier.load_weights('./' + path_models + '/disease_classifier_weights_'+ str(ee) +'.h5')
    make_trainable(disease_classifier, False)
    disease_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# obtain identity features to train anonymization VAE
load_pretrained_models(pre_trained_model_ee)
_, id_feat_train, _ = feature_extractor.predict(x_train)
_, id_feat_test, _ = feature_extractor.predict(x_test)
_, id_feat_valid, _ = feature_extractor.predict(x_valid)

###################
# Build the Model #
###################

def sampling_2(args):
    params = args
    epsilon = K.random_normal(shape=(K.shape(params)[0], identity_latent_dim), mean=0., stddev=1)
    return epsilon

def model_identity_encoder():
    input_vector = Input(shape=(latent_dim,))
    h = Dense(latent_dim // 4, activation='relu')(input_vector) # 128
    h = Dropout(0.2)(h)
    mean = Dense(identity_latent_dim)(h) # 16
    logvar = Dense(identity_latent_dim)(h) # 16
    sample = Lambda(sampling, output_shape=(identity_latent_dim,))([mean, logvar])
    features = concatenate([mean, logvar])

    encoder = Model(input_vector, [sample, features], name = 'Encoder')
    encoder.compile(loss='binary_crossentropy', optimizer=opt)
    return encoder

def model_identity_decoder():
    input_vector = Input(shape=(identity_latent_dim,))
    h = Dense(latent_dim // 4, activation='relu')(input_vector) # 128
    h = Dropout(0.2)(h)
    h = Dense(latent_dim)(h) # 32

    decoder = Model(input_vector, h, name = 'Decoder')
    decoder.compile(loss='binary_crossentropy', optimizer=opt)
    return decoder

def kl_loss(y_pred):
    z_mean = y_pred[:, 0:identity_latent_dim]
    z_log_var = y_pred[:, identity_latent_dim:identity_latent_dim*2]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)

def max_entropy_loss(y_pred):
    return K.mean(y_pred * K.log(y_pred))

encoder = model_identity_encoder()
decoder = model_identity_decoder()

d_input = Input(shape = (latent_dim,), name='identity_feat_input')
d_input2 = Input(shape = (latent_dim,), name='identity_diff_input')
decay = Input(shape = (1,), name='decay')
targets = Input(shape = (num_pp,), name='targets')

sample, params = encoder(d_input)
xpred = decoder(sample)
new_sample = Lambda(sampling_2, output_shape=(identity_latent_dim,))(params)
new_id = decoder(new_sample)

model = Model([d_input, d_input2, decay], [xpred, params])
model_loss = mean_squared_error(d_input, xpred) + decay * kl_loss(params) + \
             0.1*K.maximum(0.1 - mean_squared_error(d_input, new_id), 0) + \
             0.1*K.maximum(0.1 - mean_squared_error(d_input2, new_id), 0)
model.add_loss(model_loss)
model.compile(optimizer = opt, loss = None)
print('################### Model ###################')
model.summary()

def get_averaged_train_identities(k, n):
    identities = []
    id_indexes = []
    for i in range(0, n):
        if n < len(y_test2):
            ids = np.random.choice(np.unique(y_train1[np.where(y_train2 != y_test2[n])[0]]), k, replace=False)
        else:
            ids = np.random.choice(np.unique(y_train1), k, replace=False)
        indexes = []
        for val in ids:
            indexes.append(id_feat_train[np.where(y_train1 == val)[0][0]])
        identities.append(np.average(indexes, axis=0))
        id_indexes.append(ids)
    return np.asarray(identities), np.asarray(id_indexes)

def plot_generated_images(epoch, idx = 9, idx2=None):
    n = 2
    ind = list(range(1, n))
    ind.append(0)

    if idx2 == None:
        idx2 = idx + 4

    sample = []
    sample.append(x_test[np.where(y_test2 == 0)[0]][idx])
    sample.append(x_test[np.where(y_test2 == 1)[0]][idx2])
    sample = np.asarray(sample)

    m_feat, i_feat, o_feat = feature_extractor.predict(sample)
    new_i_feat = decoder.predict(np.random.normal(0, 1, (n, identity_latent_dim)))
    recons_i_feat = decoder.predict(encoder.predict(i_feat)[0])
    generated_images_rec = generator.predict([m_feat, i_feat, o_feat])
    #generated_images_id = generator.predict([m_feat, get_averaged_train_identities(avg_val, len(x_test))[0], o_feat])
    generated_images_id = generator.predict([m_feat, new_i_feat, o_feat])
    
    plt.figure(figsize=(12, 4))
    for i in range(n):
        for j in range(3):
        # display original
            ax = plt.subplot(2, 6, i * 3 + j + 1)
            ori = sample[i]
            ori = np.uint8(ori * 127.5 + 127.5)
            if grayscale:
                ori = ori.reshape(img_rows, img_cols)
                plt.imshow(ori, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(ori)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, 6, i * 3 + j + 7)
            if j != 1:
                rec = generated_images_rec[i]
            else:
                rec = generated_images_med[i]
            if j == 2:
                rec = generated_images_id[i] 
            rec = np.uint8(rec * 127.5 + 127.5)
            plt.imshow(rec)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # Path to be created
    plt.savefig(path + '/' + str(epoch) +'.jpg')
    plt.close()

def load_models(ee):
    encoder.load_weights('./' + path_models + '/encoder_weights_'+ str(ee) +'.h5')
    encoder.compile(loss='binary_crossentropy', optimizer=opt)
    decoder.load_weights('./' + path_models + '/decoder_weights_'+ str(ee) +'.h5')
    decoder.compile(loss='binary_crossentropy', optimizer=opt)

def save_models(ee):
    encoder.save_weights('./' + path_models + '/encoder_weights_'+ str(ee) +'.h5')
    decoder.save_weights('./' + path_models + '/decoder_weights_'+ str(ee) +'.h5')

idx_diff_val = []
for ind in range(len(x_valid)):
    poss_val = np.where(y_valid1 != y_valid1[ind])[0]
    idx_diff_val.append(random.sample(list(poss_val), 1)[0])

def train_model():
    batch_count = x_train.shape[0] / batch_size
    min_loss = 1000
    num_steps_no_improv = 0
    for ee in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % ee, '-' * 15)
        train_loss=[]
        for e in tqdm(range(int(batch_count))):
            # randomly select identity from the data to maximize distance between this identity and generated one
            idx = random.sample(range(0, x_train.shape[0]), batch_size)
            idx_diff = []
            for ind in idx:
                poss = np.where(y_train1 != y_train1[ind])[0]
                idx_diff.append(random.sample(list(poss), 1)[0])

            decay_val = 2e-3 

            g_loss = model.train_on_batch([id_feat_train[idx], id_feat_train[idx_diff], np.ones((batch_size,))*decay_val], y = None)
            train_loss.append(np.average(g_loss[0]))
        print('Train loss: ' + str(np.average(train_loss)))
        
        v_loss = model.test_on_batch([id_feat_valid, id_feat_valid[idx_diff_val], np.ones((len(id_feat_valid),))*decay_val], y=None)
        print('Validation loss: ' + str(np.average(v_loss)))
        plot_generated_images(ee, 20, 28)
        if ee > 100:
            if np.average(v_loss) < min_loss:
                min_loss = np.average(v_loss)
                save_models('anon')
                num_steps_no_improv = 0
            num_steps_no_improv += 1
            if num_steps_no_improv > 80:
                break

path = "./images"
if os.path.isdir(path) == True:
    shutil.rmtree(path)
os.mkdir(path)

train_model()
save_models('anon')

######################
##### Test model #####
######################

load_ee = 'anon'            # name given to anonymization model weights
load_models(load_ee)
idx = 20# 20   25
idx2 = 28# 28   9

with open('./test_indexes_celeba_hq.txt') as f:
    lines = f.readlines()
    test_indexes = lines[0].split(' ')
    test_indexes = np.asarray([int(i) for i in test_indexes])
    test_labels = lines[1].split(' ')
    test_labels = np.asarray([int(i) for i in test_labels])

new_id_feat = decoder.predict(np.random.normal(0, 1, (len(x_test), identity_latent_dim)))
med_feat, id_feat, oth_feat = feature_extractor.predict(x_test)

pred_dis = disease_classifier.predict(med_feat) >= 0.5
pred_dis = pred_dis.reshape((len(pred_dis),)).astype(int)

ori_indexes = np.repeat(range(0, len(id_feat)), 2)
threshold = np.average(K.eval(mean_squared_error(K.constant(id_feat[ori_indexes]), K.constant(id_feat[test_indexes]))))
threshold = 0.05

recons_ori = generator.predict([med_feat, id_feat, oth_feat])
recons_images = generator.predict([med_feat, decoder.predict(encoder.predict(id_feat)[0]), oth_feat])
print("MSE VAE: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(decoder.predict(encoder.predict(id_feat)[0])))))))
print("KL VAE: " + str(np.average(K.eval(kl_loss(K.constant(encoder.predict(id_feat)[1]))))))
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(recons_images * 127.5 + 127.5), 255)))))
print("SSIM (ori): " + str(np.average(K.eval(tf.image.ssim(K.constant(recons_ori * 127.5 + 127.5), K.constant(recons_images * 127.5 + 127.5), 255)))))

generated_images = generator.predict([med_feat, new_id_feat, oth_feat])
print('  Identity Disentanglement:')
med_feat_id, id_feat_id, oth_feat_id = feature_extractor.predict(generated_images)
print("MSE ori id: " + str(np.average(K.eval(mean_squared_error(K.constant(new_id_feat), K.constant(id_feat))))))
print("MSE ori-new id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_id))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(new_id_feat), K.constant(id_feat_id))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_id))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_id))))))
print(disease_classifier.evaluate(med_feat_id, y_test2))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_id)))
pred_id = pred_id < threshold
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))


print(disease_classifier.evaluate(med_feat_id, pred_dis))
pred_rec = disease_classifier.predict(med_feat_id) >= 0.5
print("Accuracy against predictions: " + str(accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))
print(classification_report(pred_dis.reshape((len(pred_dis),)), pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis.reshape((len(pred_dis),)), pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))


#get all individual identities:
unique_ids, unique_indexes = np.unique(y_train1, return_index=True)
mean_acc = []
num_acc = []
for identity in id_feat_id:
    rep_id_feat_id = np.tile(np.asarray([identity]), (len(unique_indexes), 1))
    pred_id = K.eval(mean_squared_error(K.constant(id_feat_train[unique_indexes]), K.constant(rep_id_feat_id)))
    pred_id = pred_id < threshold
    mean_acc.append(accuracy_score(np.ones((len(unique_indexes)),), pred_id))
    num_acc.append(len(np.where(pred_id == 1)[0]))

print('Mean Accuracy comparing to all training identities: ' + str(np.average(mean_acc)))
print('Max Accuracy comparing to all training identities: ' + str(np.max(mean_acc)))
print('Avg num where dist < threshold: ' + str(np.average(num_acc)))
print('Max num where dist < threshold: ' + str(np.max(num_acc)))

##################
##### K-SAME #####
##################

'''
med_feat, id_feat, oth_feat = feature_extractor.predict(x_test)
threshold = 0.05

avg_val = 6
plot_generated_images(avg_val, 20, 28)
avg_val = 12
plot_generated_images(avg_val, 20, 28)
avg_val = 115
plot_generated_images(avg_val, 20, 28)

avg_val = 6
averaged_ids, indexes = get_averaged_train_identities(avg_val, len(x_test))
generated_images = generator.predict([med_feat, averaged_ids, oth_feat])
print('\n\n 6 K-Same Identity Disentanglement:')
med_feat_id, id_feat_id, oth_feat_id = feature_extractor.predict(generated_images)
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(averaged_ids), K.constant(id_feat_id))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_id))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_id))))))
print(disease_classifier.evaluate(med_feat_id, y_test2))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_id)))
pred_id = pred_id < threshold
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))

unique_ids, unique_indexes = np.unique(y_train1, return_index=True)
mean_acc = []
num_acc = []
for identity in id_feat_id:
    rep_id_feat_id = np.tile(np.asarray([identity]), (len(unique_indexes), 1))
    pred_id = K.eval(mean_squared_error(K.constant(id_feat_train[unique_indexes]), K.constant(rep_id_feat_id)))
    pred_id = pred_id < threshold
    mean_acc.append(accuracy_score(np.ones((len(unique_indexes)),), pred_id))
    num_acc.append(len(np.where(pred_id == 1)[0]))

print('Mean Accuracy comparing to all training identities: ' + str(np.average(mean_acc)))
print('Max Accuracy comparing to all training identities: ' + str(np.max(mean_acc)))
print('Avg num where dist < threshold: ' + str(np.average(num_acc)))
print('Max num where dist < threshold: ' + str(np.max(num_acc)))
'''