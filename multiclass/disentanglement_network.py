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
from tensorflow.nn import l2_normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## variables ##
batch_size = 32                     # batch size
num_pp = 115                        # number of  identities
epochs = 10000                      # number of epochs
grayscale = False                   # auxiliary variable that differentiates between grayscale and rgb
opt = Adam(lr=2e-5, decay=1e-6)     # optimizer
latent_dim = 256                    # dimensions of latent vectors
crop_shape = (16, 16, 1)            # shape to crop synthetic images for augmentation purposes

#############
# Load data #
#############

f = h5py.File('../warsaw_data.hdf5')

label1 = f['id']                 # labels for the identity recognition network
label1 = np.asarray(label1)
label2 = f['dis']                # labels for the task-related classification network (glaucoma)
label2 = np.asarray(label2)
label3 = f['set']                # labels for the dataset to which the sample belongs (train - 0, test - 1 or validation - 2)
label3 = np.asarray(label3)
x = f['images']                  # image data
x = np.asarray(x)
if len(x.shape) == 3:
    grayscale = True
    x = np.reshape(x, (-1,64,64,1))

# normalize data

x = (x - 127.5) / 127.5
x = x.astype('float16')

# separate data into train, test and validation

idx_train = np.asarray(np.where(label3 == 0)[0])
idx_test  = np.asarray(np.where(label3 == 1)[0])
idx_valid  = np.asarray(np.where(label3 == 2)[0])

x_train = x[idx_train]
x_test = x[idx_test]
x_valid = x[idx_valid]
y_train1 = label1[idx_train]
y_test1 = label1[idx_test]
y_valid1 = label1[idx_valid]
y_train2 = label2[idx_train]
y_test2 = label2[idx_test]
y_valid2 = label2[idx_valid]

y_train1 = to_categorical(y_train1, num_pp)
y_valid1  = to_categorical(y_valid1, num_pp)
y_test1  = to_categorical(y_test1, num_pp)

input_shape = x_train.shape[1:]
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

# function to get template image whose medical/identity features are used to replace the ones from the original image
# on the training data, the choice of template is random, while in other sets, for each image, an image with an opposite pathology is obtained
def get_indexes(t_set, training_set = False):
    if training_set:
        return np.random.choice(range(0, len(t_set)), len(t_set), replace=False)
    indexes_no_pathology = np.where(t_set == 0)[0]
    indexes_pathology = np.where(t_set == 1)[0]
    t_indexes_pathology = np.random.choice(indexes_no_pathology, len(indexes_pathology))
    t_indexes_no_pathology = np.random.choice(indexes_pathology, len(indexes_no_pathology))
    t_indexes = np.asarray(range(0, len(t_set)))
    ind1 = np.where(np.in1d(t_indexes, indexes_pathology))
    ind2 = np.where(np.in1d(t_indexes, indexes_no_pathology))
    t_indexes[ind1] = t_indexes_pathology
    t_indexes[ind2] = t_indexes_no_pathology
    return t_indexes

# get pairs of images used on testing (saved in a txt file to ensure reproducibility)
# if pathology_indexes.txt file does not exist, one can be created with the indexes obtained through the get_indexes() function
with open('../pathology_indexes.txt') as f:
    lines = f.readlines()
    train_indexes = lines[0].split(' ')
    train_indexes = [int(i) for i in train_indexes]
    test_indexes = lines[1].split(' ')
    test_indexes = [int(i) for i in test_indexes]
    valid_indexes = lines[2].split(' ')
    valid_indexes = [int(i) for i in valid_indexes]

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

    h = Conv2D(latent_dim // 4, (3, 3), strides=(2, 2), padding='same')(d0)
    h = LeakyReLU(0.2)(h)
    
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

def model_identity_classifier(latent_dim = 256):
    input_vector = Input((latent_dim,))
    h = Dense(latent_dim // 2)(input_vector)
    h = Dropout(0.5)(h)
    h = Dense(num_pp, activation='softmax')(h)
    classifier = Model(input_vector, h, name='id_classifier')
    classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return classifier

def model_discriminator():
    d0 = Input((input_shape))

    h = Lambda(augment, output_shape=(crop_shape,))(d0)

    h = Conv2D(latent_dim // 4, (3, 3), strides=(2, 2), padding='same')(h) #8 x 8 x 32
    h = LeakyReLU(0.2)(h)

    h = res_block(h, latent_dim, resample='downsample')
    h = res_block(h, latent_dim)

    h = GlobalAveragePooling2D()(h)
    h = Dense(latent_dim)(h)
    h = Dropout(0.5)(h)
    h = Dense(latent_dim // 4)(h)
    h = Dropout(0.5)(h)
    h = Dense(1, activation='sigmoid')(h)
    classifier = Model(d0, h, name='discriminator')
    classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return classifier

###########################
#        Generator        #
###########################

def model_generator():
    units = 128
    medical_features = Input(shape=(latent_dim,))
    identity_features = Input(shape=(latent_dim,))
    other_features = Input(shape=(latent_dim,))
    input_vector = concatenate([medical_features, identity_features, other_features])
    h = Dense(4 * 4 * units, activation='relu')(input_vector)
    h = Reshape((4, 4, units))(h)
    h = LeakyReLU(0.2)(h)
    
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

##############################
# Build the Model Components #
##############################

feature_extractor = model_feature_extractor()
generator = model_generator()
identity_classifier = model_identity_classifier()
disease_classifier = model_disease_classifier()
discriminator = model_discriminator()
make_trainable(discriminator, False)

#######################
#   Build the Model   #
#######################

d_input = Input(shape = input_shape, name='net_input')
d_input2 = Input(shape = input_shape, name='net_input_2')
targets_dis = Input(shape = (1,), name='targets_dis')
targets_id = Input(shape = (num_pp,), name='targets_id')
targets_disc = Input(shape = (1,), name='targets_disc')
loss_weights = Input(shape = (1,), name='weights')

medical_features, identity_features, other_features = feature_extractor(d_input)
noisy_identity_features = GaussianNoise(0.1)(identity_features)
noisy_medical_features = GaussianNoise(0.1)(medical_features)
noisy_other_features = GaussianNoise(0.1)(other_features)
xpred = generator([noisy_medical_features, noisy_identity_features, noisy_other_features])
output_disease = disease_classifier(noisy_medical_features)
output_identity = identity_classifier(noisy_identity_features)
output_discriminator = discriminator(xpred)

altered_med_features, altered_id_features, altered_other_features = feature_extractor(d_input2)
x_pred_dis = generator([altered_med_features, identity_features, other_features]) # alter medical features
medical_features_dis, identity_features_dis, other_features_dis = feature_extractor(x_pred_dis)
x_pred_id = generator([medical_features, altered_id_features, other_features]) # alter identity features
medical_features_id, identity_features_id, other_features_id = feature_extractor(x_pred_id)
x_pred_other = generator([medical_features, identity_features, altered_other_features]) # alter other features
medical_features_other, identity_features_other, other_features_other = feature_extractor(x_pred_other)

model = Model([d_input, d_input2, targets_dis, targets_id, targets_disc, loss_weights], \
            [xpred, medical_features, identity_features, other_features, output_disease, output_identity, output_discriminator, \
            x_pred_dis, medical_features_dis, identity_features_dis, other_features_dis, x_pred_id, medical_features_id, \
            identity_features_id, other_features_id, x_pred_other, medical_features_other, identity_features_other, other_features_other])

model_loss = binary_crossentropy(targets_dis, output_disease) + \
          categorical_crossentropy(targets_id, output_identity) + \
          loss_weights * (0.1 * binary_crossentropy(targets_disc, output_discriminator) + \
          (48 - tf.image.psnr(d_input * 127.5 + 127.5, xpred * 125.5 + 127.5, 255)) / 24 + \
          (1 - tf.image.ssim(d_input * 127.5 + 127.5, xpred * 125.5 + 127.5, 255))) + \
          5 * (mean_squared_error(medical_features, medical_features_dis) + \
          mean_squared_error(identity_features, identity_features_dis) + \
          mean_squared_error(other_features, other_features_dis) + \
          mean_squared_error(medical_features, medical_features_id) + \
          mean_squared_error(identity_features, identity_features_id) + \
          mean_squared_error(other_features, other_features_id) + \
          mean_squared_error(identity_features, identity_features_other) + \
          mean_squared_error(medical_features, medical_features_other) + \
          mean_squared_error(other_features, other_features_other))
model.add_loss(model_loss)
model.compile(optimizer = opt, loss = None)
print('################### Model ###################')
model.summary()

def plot_generated_images(epoch, idx = 0):
    n = 2
    ind = list(range(1, n))
    ind.append(0)

    sample = x_test[idx:idx+n]

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

def load_all_models(ee, load_discriminator = True):
    feature_extractor.load_weights('./' + path_models + '/feature_extractor_weights_'+ str(ee) +'.h5')
    feature_extractor.compile(loss='binary_crossentropy', optimizer=opt)
    generator.load_weights('./' + path_models + '/generator_weights_'+ str(ee) +'.h5')
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    disease_classifier.load_weights('./' + path_models + '/disease_classifier_weights_'+ str(ee) +'.h5')
    disease_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    identity_classifier.load_weights('./' + path_models + '/identity_classifier_weights_'+ str(ee) +'.h5')
    identity_classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if load_discriminator:
        discriminator.load_weights('./' + path_models + '/discriminator_weights_'+ str(ee) +'.h5')
        discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

def save_all_models(ee):
    feature_extractor.save_weights('./' + path_models + '/feature_extractor_weights_'+ str(ee) +'.h5')
    generator.save_weights('./' + path_models + '/generator_weights_'+ str(ee) +'.h5')
    disease_classifier.save_weights('./' + path_models + '/disease_classifier_weights_'+ str(ee) +'.h5')
    identity_classifier.save_weights('./' + path_models + '/identity_classifier_weights_'+ str(ee) +'.h5')
    discriminator.save_weights('./' + path_models + '/discriminator_weights_'+ str(ee) +'.h5')

def train_model():
    batchCount = x_train.shape[0] / batch_size
    min_loss = 1000
    num_steps_no_improv = 0
    for ee in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % ee, '-' * 15)
        train_loss=[]
        train_dfake_loss=[]
        train_dreal_loss=[]
        train_indexes = get_indexes(y_train2, True)
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

            make_trainable(discriminator, False)
            idx = random.sample(range(0, x_train.shape[0]), batch_size)
            
            g_loss = model.train_on_batch([x_train[idx], x_train[train_indexes][idx], y_train2[idx], y_train1[idx], np.ones((batch_size, 1)), np.ones((batch_size, 1))], y = None)
            train_loss.append(np.average(g_loss[0]))
        print('Generator train loss: ' + str(np.average(train_loss)))
        print('Discriminator real loss: ' + str(np.average(train_dreal_loss)))
        print('Discriminator fake loss: ' + str(np.average(train_dfake_loss)))
        
        v_loss = model.test_on_batch([x_valid, x_valid[valid_indexes], y_valid2, y_valid1, np.ones((len(x_valid), 1)), np.zeros((len(x_valid), 1))], y=None)
        print('Validation loss: ' + str(np.average(v_loss)))
        plot_generated_images(ee, 1)
        if ee > 2500:
            if np.average(v_loss) < min_loss:
                min_loss = np.average(v_loss)
                save_all_models('eye')
                num_steps_no_improv = 0
            num_steps_no_improv += 1
            if num_steps_no_improv > 600:
                break

path = "images"
if os.path.isdir(path) == True:
    shutil.rmtree(path)
os.mkdir(path)

path_models = "models"
if os.path.isdir(path_models) == True:
    shutil.rmtree(path_models)
os.mkdir(path_models)

train_model()

load_all_models('eye')
plot_generated_images(0, 0)
plot_generated_images(1, 1)
plot_generated_images(2, 2)
plot_generated_images(4, 4)
plot_generated_images(6, 6)

######################
##### Test model #####
######################

med_feat, id_feat, oth_feat = feature_extractor.predict(x_test)
print(disease_classifier.evaluate(med_feat, y_test2))
print(identity_classifier.evaluate(id_feat, y_test1))

pred_dis = disease_classifier.predict(med_feat) >= 0.5
pred_dis = pred_dis.reshape((len(pred_dis),)).astype(int)
pred_id = identity_classifier.predict(id_feat)

print("Medical Results: " + str(accuracy_score(y_test2, pred_dis.reshape((len(pred_dis),)))))
print(classification_report(y_test2, pred_dis >= 0.5, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(y_test2, pred_dis >= 0.5))

print("Identity Results: " + str(accuracy_score(np.argmax(y_test1, axis=-1), np.argmax(pred_id, axis=-1))))
print('Identity Recognition on Target Identities: ' + str(accuracy_score(np.argmax(y_test1[test_indexes], axis=-1), np.argmax(pred_id, axis=-1))))


generated_images = generator.predict([med_feat, id_feat, oth_feat])
print("MSE: " + str(np.average(K.eval(mean_squared_error(K.constant(x_test), K.constant(generated_images))))))
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images * 127.5 + 127.5), 255)))))
print("PSNR: " + str(np.average(K.eval(tf.image.psnr(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images * 127.5 + 127.5), 255)))))

print('Reconstruction')
rec_med_feat, rec_id_feat, rec_oth_feat = feature_extractor.predict(generated_images)
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(rec_med_feat))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(rec_id_feat))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(rec_oth_feat))))))
print('Medical Results')
print(disease_classifier.evaluate(rec_med_feat, y_test2))
print(confusion_matrix(y_test2, disease_classifier.predict(rec_med_feat) >= 0.5))
print('Identity Results')
print(identity_classifier.evaluate(rec_id_feat, y_test1))
print(identity_classifier.evaluate(rec_id_feat, y_test1[test_indexes]))

pred_rec = disease_classifier.predict(rec_med_feat) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against predictions: " + str(accuracy_score(pred_dis, pred_rec)))
print(classification_report(pred_dis, pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis, pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis, pred_rec)))


print('  Medical Disentanglement:')
altered_med_feat = med_feat[test_indexes]
generated_images_med = generator.predict([altered_med_feat, id_feat, oth_feat])
med_feat_med, id_feat_med, oth_feat_med = feature_extractor.predict(generated_images_med)
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images_med * 127.5 + 127.5), 255)))))
print("MSE ori med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(altered_med_feat))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_med_feat), K.constant(med_feat_med))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_med))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_med))))))
print(disease_classifier.evaluate(med_feat_med, y_test2[test_indexes]))
print(confusion_matrix(y_test2[test_indexes], disease_classifier.predict(med_feat_med) >= 0.5))
print(identity_classifier.evaluate(id_feat_med, y_test1))
print(identity_classifier.evaluate(id_feat_med, y_test1[test_indexes]))

pred_rec = disease_classifier.predict(med_feat_med) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against predictions: " + str(accuracy_score(pred_dis, pred_rec)))
print(classification_report(pred_dis, pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis, pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis, pred_rec)))

print('  Identity Disentanglement:')
altered_id_feat = id_feat[test_indexes]
generated_images_id = generator.predict([med_feat, altered_id_feat, oth_feat])
med_feat_id, id_feat_id, oth_feat_id = feature_extractor.predict(generated_images_id)
print("SSIM: " + str(np.average(K.eval(tf.image.ssim(K.constant(x_test * 127.5 + 127.5), K.constant(generated_images_id * 127.5 + 127.5), 255)))))
print("MSE ori id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(altered_id_feat))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_id_feat), K.constant(id_feat_id))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_id))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(oth_feat_id))))))
print(disease_classifier.evaluate(med_feat_id, y_test2))
print(confusion_matrix(y_test2[test_indexes], disease_classifier.predict(med_feat_id) >= 0.5))
print(identity_classifier.evaluate(id_feat_id, y_test1[test_indexes]))
print(identity_classifier.evaluate(id_feat_id, y_test1))

pred_rec = disease_classifier.predict(med_feat_id) >= 0.5
pred_rec = pred_rec.reshape((len(pred_rec,))).astype(int)
print("Accuracy against predictions: " + str(accuracy_score(pred_dis, pred_rec)))
print(classification_report(pred_dis, pred_rec, target_names=['not glaucoma', 'glaucoma']))
print(confusion_matrix(pred_dis, pred_rec))
print('Change of prediction: ' + str(1-accuracy_score(pred_dis, pred_rec)))

print('  Other Features Disentanglement:')
altered_oth_feat = oth_feat[test_indexes]
generated_images_oth = generator.predict([med_feat, id_feat, altered_oth_feat])
med_feat_oth, id_feat_oth, oth_feat_oth = feature_extractor.predict(generated_images_oth)
print("MSE ori other: " + str(np.average(K.eval(mean_squared_error(K.constant(oth_feat), K.constant(altered_oth_feat))))))
print("MSE other: " + str(np.average(K.eval(mean_squared_error(K.constant(altered_oth_feat), K.constant(oth_feat_oth))))))
print("MSE id: " + str(np.average(K.eval(mean_squared_error(K.constant(id_feat), K.constant(id_feat_oth))))))
print("MSE med: " + str(np.average(K.eval(mean_squared_error(K.constant(med_feat), K.constant(med_feat_oth))))))
print(disease_classifier.evaluate(med_feat_oth, y_test2))
print(identity_classifier.evaluate(id_feat_oth, y_test1))
print(identity_classifier.evaluate(id_feat_oth, y_test1[test_indexes]))

test_indexes_2 = []
for ind in range(len(x_test)):
    poss_val = np.where(label1[idx_test] != label1[idx_test][ind])[0]
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
