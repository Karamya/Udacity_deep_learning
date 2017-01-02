# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:25:09 2016

@author: Karthick Perumal
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from scipy.fftpack import dct
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import time
#########################################################################################################

### Download the compressed -tar-gz file

#########################################################################################################


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    """ A hook to report the progress of a download. This is mostly intended for users 
    with slow internet connections. reports every 1% change in donwload progress
    """
    global last_percent_reported
    percent = int(count*blockSize*100/totalSize)
    
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
            
        last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force = False):
    """ Download a file if not present, and make sure it's the right size"""
    if force or not os.path.exists(filename):
        print('Attempting to download: ', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook = download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename
    
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


#########################################################################################################

### Extract the dataset from the compressed -tar-gz file

#########################################################################################################

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force = False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  ## remove .tar.gz
    if os.path.isdir(root) and not force:
        ### you may override by setting force = True
        print('%s already present - Skipping extraction of %s.' %(root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' %(num_classes, len(data_folders)))
    print(data_folders)
    return data_folders
    
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


#########################################################################################################

### Peek at some of the data

#########################################################################################################

image_size = 28  #pixel width and height
pixel_depth = 255.0 # number of levels per pixel

def load_letter(folder, min_num_images):
    """Load the data for a single letter label"""
    image_files = os.listdir(folder)
    #print(image_files)
    dataset = np.ndarray(shape = (len(image_files), image_size, image_size), dtype = np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' %str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read: ', image_file, ':', e, '- it\'s ok, skipping.')
        
    dataset = dataset[0: num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Fewer images than expected: %d < %d' % (num_images, min_num_images))
        
    print('Full dataset tensor: ', dataset.shape)
    print('Mean: ', np.mean(dataset))
    print('Standard deviation: ', np.std(dataset))
    return dataset
    
def maybe_pickle(data_folders, min_num_images_per_class, force = False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            ## You may override by setting force = True
            print('%s already present - Skipping pickling.' %set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names
    
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

#########################################################################################################

### Problem 2: Display a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot

#########################################################################################################
"""
import random
import glob2
def display_sample():
    print("   -----   ")    
    fig = plt.figure(1)
    for i, pickle_file in enumerate(sorted(glob2.glob('D:/Data analysis/udacity/notMNIST_small/*.pickle'))):
        with open(pickle_file, 'rb') as p:
            data = pickle.load(p)
            print('Shape of {0} is {1}'.format(str(p)[-10:-2], data.shape))
            a = fig.add_subplot(2, 5, i+1)
            plt.imshow(data[random.randrange(len(data))])
            a.set_title(str(p)[-10:-9])
    plt.show()
    plt.suptitle('Randomized data')
display_sample()

#########################################################################################################

### Problem 3: Another check: we expect the data to be balanced across classes. Verify that.

#########################################################################################################
for pickle_file in sorted(glob2.glob('D:/Data analysis/udacity/notMNIST_small/*.pickle')):
    with open(pickle_file, 'rb') as p:
        data = pickle.load(p)
        print('{0} test data count is {1}'.format(str(p)[-10:-2], data.shape))
for pickle_file in sorted(glob2.glob('D:/Data analysis/udacity/notMNIST_large/*.pickle')):
    with open(pickle_file, 'rb') as p:
        data = pickle.load(p)
        print('{0} train data count is {1}'.format(str(p)[-10:-2], data.shape))
#########################

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype = np.float32)
        labels = np.ndarray(nb_rows, dtype = np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels
    
def merge_datasets(pickle_files, train_size, valid_size = 0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                ## Let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
                
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels
    
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training: ', train_dataset.shape, train_labels.shape)
print('Validation: ', valid_dataset.shape, valid_labels.shape)
print('Testing: ', test_dataset.shape, test_labels.shape)
            
### Next we'll randomize the data- It's important to have the labels well shuffled for the training and test distributions to match
            
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
    
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#########################################################################################################

### Problem 4: Convince yourself that the data is still good after shuffling

#########################################################################################################
fig = plt.figure(2)
for i in range(10):
    fig.add_subplot(2, 5, i+1)
    plt.imshow(train_dataset[random.randint(i*20000, (i +1)*20000)])
plt.show()
plt.suptitle('Randomized data')


pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to ', pickle_file, ':', e)
    raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size: ', statinfo.st_size)
"""
#########################################################################################################

### Problem 5: Convince yourself that the data is still good after shuffling

#########################################################################################################
"""
By construction, this dataset might contain a lot of overlapping samples, including training data 
that's also contained in the validation and test set! Overlap between training and test can skew 
the results if you expect to use your model in an environment where there is never an overlap, 
but are actually ok if you expect to see training samples recur when you use it. Measure how much 
overlap there is between training, validation and test samples.

Optional questions:
What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
"""
"""
from hashlib import md5 as m

#def count_exact_duplicates(dataset1, dataset2):
#    print(len(dataset1), len(dataset2))
#    hashes_1 =  set(m(img).hexdigest() for img in dataset1)
#    hashes_2 = set(m(img).hexdigest() for img in dataset2)
#    overlap = {}
#    for i, hash1 in enumerate(hashes_1):
#        for j, hash2 in enumerate(hashes_2):
#            if hash1== hash2:
#                if not i in overlap.keys():
#                    overlap[i] = []
#                overlap[i].append(j)
#                #plt.imshow(hash1)
#                #break
#    print(len(overlap))
#    return
# 
def count_exact_duplicates1(dataset1, dataset2):  ### Same function as above, but to display the matches
    print(len(dataset1), len(dataset2))
    overlap = {}
    start_time = time.time()
    for i, img1 in enumerate(dataset1):
        for j, img2 in enumerate(dataset2):
            if m(img1).hexdigest()== m(img2).hexdigest():
                if not i in overlap.keys():
                    overlap[i] = []
                overlap[i].append(j)
#                print(i, j)
#                fig = plt.figure()                ### Uncomment to see an example display
#                fig.add_subplot(1,2,1)
#                plt.imshow(img1)
#                fig.add_subplot(1,2,2)
#                plt.imshow(img2)
#                return
    print("Loop Executed in " +str( (time.time() - start_time)) + " seconds..." )         
    print(len(overlap))
    return   
    
with open('notMNIST.pickle', 'rb') as p:
    pd = pickle.load(p)
    count_exact_duplicates1(pd['test_dataset'],  pd['train_dataset'])  ##1153   ## 1 
    count_exact_duplicates1(pd['valid_dataset'], pd['train_dataset'])  ## 953   ## 1 
    count_exact_duplicates1(pd['test_dataset'],  pd['valid_dataset'])  ## 55    ## 1 200

################## Problem 5: Optional - almost identical  ###################  
  
import imagehash  

from PIL import Image as im
def count_almost_duplicates(dataset1, dataset2):
    overlap = {}
    #print(time.time())
    start_time = time.time()
    for i, img1 in enumerate(dataset1):
        for j, img2 in enumerate(dataset2):           
            h1 = imagehash.phash(im.fromarray(img1, mode = 'L'))
            h2 = imagehash.phash(im.fromarray(img2, mode = 'L'))
            if ((h1-h2)/len(h1.hash)**2) < 0.05:
                if not i in overlap.keys():
                    overlap[i] = []
                overlap[i].append(j)
                fig = plt.figure()                ### Uncomment to see an example display
                fig.add_subplot(1,2,1)
                plt.imshow(img1)
                fig.add_subplot(1,2,2)
                plt.imshow(img2)
               
    print(len(overlap)) 
    print("Loop Executed in " +str( (time.time() - start_time)) + " seconds..." ) 
    return

with open('notMNIST.pickle', 'rb') as p:
    pd = pickle.load(p)
    count_almost_duplicates(pd['test_dataset'],  pd['valid_dataset'])
    count_almost_duplicates(pd['test_dataset'],  pd['train_dataset'])  ##1153   ## 1 
    count_almost_duplicates(pd['valid_dataset'], pd['train_dataset'])
    #print(imagehash.average_hash(pd['test_dataset'][11]))
""" 
#################### Probelm 5: Optional -sanitize   
#def sanitized()

#########################################################################################################

### Problem 6: Train a simple model on this data using 50, 100, 1000 and 5000 training samples. 
### Hint : You can use the logistic regression model from sklearn.linear_model

#########################################################################################################

with open('notMNIST.pickle', 'rb') as p:
    pd = pickle.load(p)
    train_dataset = pd['train_dataset']
    train_labels = pd['train_labels']
    test_dataset = pd['test_dataset']
    test_labels = pd['test_labels']
print(train_dataset.shape)
print(train_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)

model = LogisticRegression()
X_test = test_dataset.reshape(test_dataset.shape[0], 28 * 28)
y_test = test_labels

def apply_log_reg(sample_size):
    labels = {0: 'A', 1:'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}    
    start_time = time.time()
    X_train = train_dataset[:sample_size].reshape(sample_size, 28 * 28)
    y_train = train_labels[:sample_size]
    model.fit(X_train, y_train)
    output = model.score(X_test, y_test)
    print("For sample size of {}, the accuracy is {} and executed in {}".format(sample_size, output,  time.time()-start_time))
    #print(output)
    predicted_labels = model.predict(X_train)
    #print(predicted_labels)
    fig = plt.figure()
    for i in range(10):
        fig.add_subplot(2, 5, i+1)
        plt.imshow(X_train[i].reshape(28, 28))
        plt.title("Predicted {}".format(labels[predicted_labels[i]]))
        plt.axis('off')
    plt.show()
    return  


apply_log_reg(10)
apply_log_reg(100)
apply_log_reg(1000)
apply_log_reg(5000)
