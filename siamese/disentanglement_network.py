#########################
## Load trained models ##
#########################

import os, random, h5py, numpy as np, math, matplotlib, tensorflow as tf, shutil
os.environ["KERAS_BACKEND"] = "tensorflow"
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Lambda, Input, Reshape, Dense, Dropout, Activation, concatenate, Add, GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.losses import cosine_distance
from tensorflow.nn import l2_normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## variables ##
batch_size = 32                     # batch size
epochs = 1600                       # number of epochs
opt = Adam(lr=2e-5, decay=1e-6)     # gan optimizer
dopt = Adam(lr=2e-5, decay=1e-6)    # discriminator optimizer
latent_dim = 512                    # dimensions of latent vectors
crop_shape = (64, 64, 3)            # shape to crop synthetic images for augmentation purposes
grayscale = False                   # auxiliary variable that differentiates between grayscale and rgb

#############
# Load data #
#############
f = h5py.File('celeba_hq.hdf5')

label1 = f['ids']                 # labels for the identity recognition network
label1 = np.asarray(label1)
label2 = f['atts']                # labels for the task-related classification network (smile detection)
label2 = np.asarray(label2)[:, 31]
label3 = f['set']                # labels for the dataset to which the sample belongs (train - 0, test - 1 or validation - 2)
label3 = np.asarray(label3)
x = f['images']                  # image data
x = np.asarray(x)
f.close()
if len(x.shape) == 3:
    grayscale = True
    x = np.reshape(x, (-1, 64, 64, 1))

# normalize image data between -1 and 1
x = (x - 127.5) / 127.5
x = x.astype('float16')

# split data in train, test and validation sets
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

# augment synthetic images before providing them to discriminator to deal with limited data
def augment(img):
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.random_crop(img, [tf.shape(img)[0], crop_shape[0], crop_shape[1], crop_shape[2]])
    return img

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

def model_discriminator():
    d0 = Input((input_shape))

    h = Lambda(augment, output_shape=(crop_shape,))(d0)

    h = Conv2D(latent_dim // 4, (3, 3), strides=(2, 2), padding='same')(h)
    h = LeakyReLU(0.2)(h)

    h = res_block(h, latent_dim // 2, resample='downsample')
    h = res_block(h, latent_dim // 2)

    h = res_block(h, latent_dim, resample='downsample') 
    h = res_block(h, latent_dim)

    h = GlobalAveragePooling2D()(h)
    h = Dense(latent_dim)(h)
    h = Dropout(0.5)(h)
    h = Dense(latent_dim // 4)(h)
    h = Dropout(0.5)(h)
    h = Dense(1, activation='sigmoid')(h)
    classifier = Model(d0, h, name='discriminator')
    classifier.compile(loss='binary_crossentropy', optimizer=dopt, metrics=['accuracy'])
    return classifier

###########################
#        Generator        #
###########################

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

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

###########################
# Build the Discriminator #
###########################

feature_extractor = model_feature_extractor()
generator = model_generator()
disease_classifier = model_disease_classifier()
discriminator = model_discriminator()

def load_all_models(ee, load_discriminator = True):
    feature_extractor.load_weights('./' + path_models + '/feature_extractor_weights_'+ str(ee) +'.h5')
    feature_extractor.compile(loss='binary_crossentropy', optimizer=opt)
    generator.load_weights('./' + path_models + '/generator_weights_'+ str(ee) +'.h5')
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    disease_classifier.load_weights('./' + path_models + '/disease_classifier_weights_'+ str(ee) +'.h5')
    disease_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    if load_discriminator:
        discriminator.load_weights('./' + path_models + '/discriminator_weights_'+ str(ee) +'.h5')
        discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

def save_all_models(ee):
    feature_extractor.save_weights('./' + path_models + '/feature_extractor_weights_'+ str(ee) +'.h5')
    generator.save_weights('./' + path_models + '/generator_weights_'+ str(ee) +'.h5')
    disease_classifier.save_weights('./' + path_models + '/disease_classifier_weights_'+ str(ee) +'.h5')
    discriminator.save_weights('./' + path_models + '/discriminator_weights_'+ str(ee) +'.h5')

make_trainable(discriminator, False)

d_input = Input(shape = input_shape, name='net_input')
d_input_id_same = Input(shape = input_shape, name='net_input_id_same')
d_input_id_diff = Input(shape = input_shape, name='net_input_id_diff')
d_input2 = Input(shape = input_shape, name='net_input_2')
targets_dis = Input(shape = (1,), name='targets_dis')
targets_id = Input(shape = (1,), name='targets_id')
targets_disc = Input(shape = (1,), name='targets_disc')
loss_weights = Input(shape = (1,), name='weights')

#######################
#   Build the Model   #
#######################

medical_features, identity_features, other_features = feature_extractor(d_input)
noisy_identity_features = GaussianNoise(0.1)(identity_features)
noisy_medical_features = GaussianNoise(0.1)(medical_features)
noisy_other_features = GaussianNoise(0.1)(other_features)
xpred = generator([noisy_medical_features, noisy_identity_features, noisy_other_features])
output_disease = disease_classifier(medical_features)
output_discriminator = discriminator(xpred)

_, same_id_features, _ = feature_extractor(d_input_id_same)
_, diff_id_features, _ = feature_extractor(d_input_id_diff)

altered_med_features, altered_id_features, altered_other_features = feature_extractor(d_input2)

x_pred_dis = generator([altered_med_features, identity_features, other_features]) # alter medical features
medical_features_dis, identity_features_dis, other_features_dis = feature_extractor(x_pred_dis)
x_pred_id = generator([medical_features, altered_id_features, other_features]) # alter identity features
medical_features_id, identity_features_id, other_features_id = feature_extractor(x_pred_id)
x_pred_other = generator([medical_features, identity_features, altered_other_features]) # alter other features
medical_features_other, identity_features_other, other_features_other = feature_extractor(x_pred_other)

model = Model([d_input, d_input2, d_input_id_same, d_input_id_diff, targets_dis, targets_id, targets_disc, loss_weights], \
            [xpred, medical_features, identity_features, other_features, output_disease, output_discriminator, \
            x_pred_dis, medical_features_dis, identity_features_dis, other_features_dis, x_pred_id, medical_features_id, \
            identity_features_id, other_features_id, x_pred_other, medical_features_other, identity_features_other, other_features_other, \
            altered_med_features, altered_id_features, altered_other_features, same_id_features, diff_id_features])

model_loss = 0.5 * binary_crossentropy(targets_dis, output_disease) + \
          10 * (mean_squared_error(identity_features, same_id_features) + \
          K.maximum(0.1 - mean_squared_error(identity_features, diff_id_features), 0)) + \
          loss_weights * binary_crossentropy(targets_disc, output_discriminator) + \
          (48 - tf.image.psnr(d_input * 127.5 + 127.5, xpred * 127.5 + 127.5, 255)) / 24 + \
          10 * (1 - tf.image.ssim(d_input * 127.5 + 127.5, xpred * 127.5 + 127.5, 255)) + \
          10 * (mean_squared_error(identity_features, identity_features_dis) + \
          mean_squared_error(other_features, other_features_dis) + \
          mean_squared_error(medical_features, medical_features_id) + \
          mean_squared_error(other_features, other_features_id) + \
          mean_squared_error(identity_features, identity_features_other) + \
          mean_squared_error(medical_features, medical_features_other)  + \
          mean_squared_error(altered_id_features, identity_features_id) + \
          mean_squared_error(altered_med_features, medical_features_dis) + \
          mean_squared_error(altered_other_features, other_features_other))

model.add_loss(model_loss)
model.compile(optimizer = opt, loss = None)
print('################### Model ###################')
model.summary()

def plot_generated_images(epoch, idx = 0, idx2 = None):
    n = 2
    ind = list(range(1, n))
    ind.append(0)

    if idx2 is None:
        idx2 = idx+4

    sample = []
    sample.append(x_test[np.where(y_test2 == 0)[0]][idx2])
    sample.append(x_test[np.where(y_test2 == 1)[0]][idx])
    sample = np.asarray(sample)

    m_feat, i_feat, o_feat = feature_extractor.predict(sample)
    generated_images_rec = generator.predict([m_feat, i_feat, o_feat])
    generated_images_med = generator.predict([m_feat[ind], i_feat, o_feat])
    generated_images_id = generator.predict([m_feat, i_feat[ind], o_feat])

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
            if j == 0:
                rec = generated_images_rec[i]
            elif j == 1:
                rec = generated_images_med[i]
            else:
                rec = generated_images_id[i]
            rec = np.uint8(rec * 127.5 + 127.5)
            if grayscale:
                rec = rec.reshape(img_rows, img_cols)
                plt.imshow(rec, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(rec)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # Path to be created
    plt.savefig(path + '/' + str(epoch) +'.jpg')
    plt.close()

idx2_val = []
idx_id_val = []
idx_diff_val = []
for ind in range(len(x_valid)):
    same_id_images = np.where(y_valid1 == y_valid1[ind])[0]
    if len(same_id_images) > 1:
        same_id_images = np.delete(same_id_images, np.where(same_id_images == ind))
    idx_id_val.append(random.sample(list(same_id_images), 1)[0])
    diff_atts = np.where(y_valid2 != y_valid2[ind])[0]
    diff_atts = np.delete(diff_atts, np.where(y_valid1[diff_atts] == y_valid1[ind]))
    idx2_val.append(random.sample(list(diff_atts), 1)[0])
    diff_atts = np.where(y_valid1 != y_valid1[ind])[0]
    idx_diff_val.append(random.sample(list(diff_atts), 1)[0])

def train_model():
    batchCount = x_train.shape[0] / batch_size
    min_loss = 1000
    num_steps_no_improv = 0
    for ee in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % ee, '-' * 15)
        train_loss=[]
        train_dfake_loss=[]
        train_dreal_loss=[]
        for e in tqdm(range(int(batchCount))):
            
            y0_dist_real = np.ones((batch_size, 1))
            y0_dist_fake = np.zeros((batch_size, 1))
            
            n_discriminator = 1
            make_trainable(discriminator, True)
            for _ in range(n_discriminator):
                idx = random.sample(range(0, x_train.shape[0]), batch_size)
                [med_feat, id_feat, oth_feat] = feature_extractor.predict(x_train[idx])
                generated_images = generator.predict([med_feat, id_feat, oth_feat])

                # Train discriminator with real images
                d_loss_real = discriminator.train_on_batch(x_train[idx], y=y0_dist_real)
                train_dreal_loss.append(np.average(d_loss_real[0]))
                
                # Train discriminator with fake images
                d_loss_fake = discriminator.train_on_batch(generated_images, y=y0_dist_fake)
                train_dfake_loss.append(np.average(d_loss_fake[0]))
            
            idx = random.sample(range(0, x_train.shape[0]), batch_size)
            idx2 = []
            idx_id = []
            idx_diff = []
            # select image pairs
            for ind in idx:
                #diff id + diff gender, diff id same gender, same id
                poss = np.where(y_train1 == y_train1[ind])[0]
                poss = np.delete(poss, np.where(poss == ind))
                idx_id.append(random.sample(list(poss), 1)[0]) # same_id
                poss = np.where(y_train2 != y_train2[ind])[0]
                poss = np.delete(poss, np.where(y_train1[poss] == y_train1[ind]))
                idx2.append(random.sample(list(poss), 1)[0]) #diff id + diff gender
                poss = np.where(y_train1 != y_train1[ind])[0]
                idx_diff.append(random.sample(list(poss), 1)[0]) #diff id 

            make_trainable(discriminator, False)
            l_weights = np.ones((batch_size, 1)) * 0.02
            g_loss = model.train_on_batch([x_train[idx], x_train[idx2], x_train[idx_id], x_train[idx_diff], y_train2[idx], np.ones((batch_size, 1)), np.ones((batch_size, 1)), l_weights], y = None)
            train_loss.append(np.average(g_loss[0]))
        print('Generator train loss: ' + str(np.average(train_loss)))
        print('Discriminator real loss: ' + str(np.average(train_dreal_loss)))
        print('Discriminator fake loss: ' + str(np.average(train_dfake_loss)))

        v_loss = model.test_on_batch([x_valid, x_valid[idx2_val], x_valid[idx_id_val], x_valid[idx_diff_val], y_valid2, np.ones((len(x_valid), 1)), np.ones((len(x_valid), 1)), np.zeros((len(x_valid), 1))], y=None)
        print('Validation loss: ' + str(np.average(v_loss)))
        plot_generated_images(ee, 28, 20)
        if ee == 500 or ee == 1000:
            min_loss = np.average(v_loss)
            num_steps_no_improv = 0
            save_all_models('model_name')
        if ee > 1500:
            if np.average(v_loss) < min_loss:
                min_loss = np.average(v_loss)
                save_all_models('model_name')
                num_steps_no_improv = 0
            num_steps_no_improv += 1
            if num_steps_no_improv > 200:
                break

path = "./images"
if os.path.isdir(path) == True:
    shutil.rmtree(path)
os.mkdir(path)

path_models = "models"
if os.path.isdir(path_models) != True:
    os.mkdir(path_models)

train_model()

######################
##### Test model #####
######################

load_all_models('model_name')

# get image pairs for testing disentanglement network
test_indexes = []
test_labels = []
for ind in range(len(x_test)):
    poss_val = np.where(y_test1 == y_test1[ind])[0]
    poss_val = np.delete(poss_val, np.where(poss_val == ind))
    test_indexes.append(random.sample(list(poss_val), 1)[0])
    test_labels.append(1)
    poss_val = np.where(y_test2 != y_test2[ind])[0]
    poss_val = np.delete(poss_val, np.where(y_test1[poss_val] == y_test1[ind]))
    test_indexes.append(random.sample(list(poss_val), 1)[0])
    test_labels.append(0)

test_indexes = np.asarray(test_indexes)
test_labels = np.asarray(test_labels)

# write to/read from file the testing image pairs
'''with open('./test_indexes_celeba_hq.txt') as f:
    
    #f.write(' '.join(list(map(str, test_indexes))))
    #f.write('\n')
    #f.write(' '.join(list(map(str, test_labels))))
    
    lines = f.readlines()
    test_indexes = lines[0].split(' ')
    test_indexes = np.asarray([int(i) for i in test_indexes])
    test_labels = lines[1].split(' ')
    test_labels = np.asarray([int(i) for i in test_labels])
'''

med_feat, id_feat, oth_feat = feature_extractor.predict(x_test)

print(disease_classifier.evaluate(med_feat, y_test2))
pred_dis = disease_classifier.predict(med_feat) >= 0.5
pred_dis = pred_dis.reshape((len(pred_dis,))).astype(int)
print("Medical Results: " + str(accuracy_score(y_test2, pred_dis.reshape((len(pred_dis),)))))
print(classification_report(y_test2, pred_dis, target_names=['female', 'male']))
print(confusion_matrix(y_test2, pred_dis))

ori_indexes = np.repeat(range(0, len(id_feat)), 2)

avg_id_dist = np.average(K.eval(mean_squared_error(K.constant(id_feat[ori_indexes]), K.constant(id_feat[test_indexes]))))
avg_id_dist = 0.05
pred_id = K.eval(mean_squared_error(K.constant(id_feat[ori_indexes]), K.constant(id_feat[test_indexes])))
pred_id = pred_id < avg_id_dist
print('Identity Recognition: ' + str(accuracy_score(test_labels, pred_id)))

target_indexes = np.where(test_labels == 0)[0]
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat[test_indexes[target_indexes]])))
pred_id = pred_id < avg_id_dist
print('Identity Recognition on Target Identities: ' + str(accuracy_score(1-test_labels[target_indexes], pred_id)))


generated_images = generator.predict([med_feat, id_feat, oth_feat])
print("MSE: " + str(np.average(K.eval(mean_squared_error(K.constant(x_test), K.constant(generated_images))))))
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images * 127.5 + 127.5), 255)))))
print("PSNR: " + str(np.average(K.eval(tf.image.psnr(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images * 127.5 + 127.5), 255)))))

print('Reconstruction')
rec_med_feat, rec_id_feat, rec_oth_feat = feature_extractor.predict(generated_images)
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(rec_med_feat))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(rec_id_feat))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(rec_oth_feat))))))
pred_rec = disease_classifier.predict(rec_med_feat) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against ground-truth: " + str(accuracy_score(y_test2, pred_rec.reshape((len(pred_rec),)))))
print(classification_report(y_test2, pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(y_test2, pred_rec))
print("Accuracy against predictions: " + str(accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))
print(classification_report(pred_dis.reshape((len(pred_dis),)), pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis.reshape((len(pred_dis),)), pred_rec))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(rec_id_feat)))
pred_id = pred_id < avg_id_dist
print('Change of prediction: ' + str(1-accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))


altered_indexes = np.where(test_labels == 0)[0]
altered_med_feat = med_feat[test_indexes[altered_indexes]]
altered_id_feat = id_feat[test_indexes[altered_indexes]]

pred_id = K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(rec_id_feat)))
pred_id = pred_id < avg_id_dist
print("Identity Acc other id: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))


print('  Medical Disentanglement:')
generated_images_med = generator.predict([altered_med_feat, id_feat, oth_feat])
med_feat_med, id_feat_med, oth_feat_med = feature_extractor.predict(generated_images_med)
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images_med * 127.5 + 127.5), 255)))))
print("MSE ori med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(altered_med_feat))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_med_feat), K.constant(med_feat_med))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_med))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_med))))))
print(disease_classifier.evaluate(med_feat_med, y_test2[test_indexes[altered_indexes]]))
print(confusion_matrix(y_test2[test_indexes[altered_indexes]], disease_classifier.predict(med_feat_med) >= 0.5))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_med)))
pred_id = pred_id < avg_id_dist
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))
pred_id = K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(id_feat_med)))
pred_id = pred_id < avg_id_dist
print("Identity Acc other id: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))

pred_rec = disease_classifier.predict(med_feat_med) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against predictions: " + str(accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))
print(classification_report(pred_dis.reshape((len(pred_dis),)), pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis.reshape((len(pred_dis),)), pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))

print('GT: ' + str(disease_classifier.evaluate(med_feat_med, y_test2)))
print(y_test2[test_indexes[altered_indexes]])
print(y_test2)
print(pred_dis.reshape((len(pred_dis),)))
print(pred_rec.reshape((len(pred_rec),)))

print('  Identity Disentanglement:')
generated_images_id = generator.predict([med_feat, altered_id_feat, oth_feat])
med_feat_id, id_feat_id, oth_feat_id = feature_extractor.predict(generated_images_id)
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images_id * 127.5 + 127.5), 255)))))
print("MSE ori id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(altered_id_feat))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(id_feat_id))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_id))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_id))))))
print(disease_classifier.evaluate(med_feat_id, y_test2))
print(confusion_matrix(y_test2, disease_classifier.predict(med_feat_id) >= 0.5))
pred_id = K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(id_feat_id)))
pred_id = pred_id < avg_id_dist
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_id)))
pred_id = pred_id < avg_id_dist
print("Identity Acc ori id: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))

pred_rec = disease_classifier.predict(med_feat_id) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against predictions: " + str(accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))
print(classification_report(pred_dis.reshape((len(pred_dis),)), pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis.reshape((len(pred_dis),)), pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis.reshape((len(pred_dis),)), pred_rec.reshape((len(pred_rec),)))))

print('  Other Features Disentanglement:')
altered_oth_feat = oth_feat[test_indexes[altered_indexes]]
generated_images_oth = generator.predict([med_feat, id_feat, altered_oth_feat])
med_feat_oth, id_feat_oth, oth_feat_oth = feature_extractor.predict(generated_images_oth)
print("MSE ori other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(altered_oth_feat))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_oth_feat), K.constant(oth_feat_oth))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_oth))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_oth))))))
print(disease_classifier.evaluate(med_feat_oth, y_test2))
pred_id = K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_oth)))
pred_id = pred_id < avg_id_dist
print("Identity Acc: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))
pred_id = K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(id_feat_oth)))
pred_id = pred_id < avg_id_dist
print("Identity Acc other id: " + str(accuracy_score(np.ones((len(x_test)),), pred_id)))

test_indexes_2 = []
for ind in range(len(x_test)):
    poss_val = np.where(y_test1 != y_test1[ind])[0]
    test_indexes_2.append(random.sample(list(poss_val), 1)[0])

med_feat_tar1, id_feat_tar1, oth_feat_tar1 = feature_extractor.predict(x_test[test_indexes_2])

gen_med_1 = generator.predict([med_feat_tar1, id_feat, oth_feat])
med_feat_tar2, id_feat_tar2, oth_feat_tar2 = feature_extractor.predict(gen_med_1)
double_gen_med = generator.predict([altered_med_feat, id_feat_tar2, oth_feat_tar2])
print("\nSSIM MR: " + str(np.average(K.eval(tf.image.ssim(K.constant(generated_images_med * 127.5 + 127.5), K.constant(double_gen_med * 127.5 + 127.5), 255)))))

id_med_1 = generator.predict([med_feat, id_feat_tar1, oth_feat])
med_feat_tar3, id_feat_tar3, oth_feat_tar3 = feature_extractor.predict(id_med_1)
double_gen_id = generator.predict([med_feat_tar3, altered_id_feat, oth_feat_tar3])
print("SSIM IR: " + str(np.average(K.eval(tf.image.ssim(K.constant(generated_images_id * 127.5 + 127.5), K.constant(double_gen_id * 127.5 + 127.5), 255)))))

med_feat_tar4, id_feat_tar4, oth_feat_tar4 = feature_extractor.predict(generated_images)
double_gen_rec = generator.predict([med_feat_tar4, id_feat_tar4, oth_feat_tar4])
print("SSIM Rec: " + str(np.average(K.eval(tf.image.ssim(K.constant(generated_images * 127.5 + 127.5), K.constant(double_gen_rec * 127.5 + 127.5), 255)))))
