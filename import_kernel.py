# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.
import bson                       # this is installed with the pymongo package
from skimage.data import imread   # or, whatever image library you prefer
import io
import os
import random
import matplotlib.pyplot as plt

class ImageData():
    """
    This object contains the dummy, train and test data of the CDiscount dataset.
    """
    def __init__(self, path='..\\datasets\\CDiscount'):
        # Initialize the datasets to be empty.
        self.dummy = None
        self.train = None
        self.test = None
        
        self.path = path
        
    def get_data(self, data_type='dummy', override=False):
        """
        This method is an interface for the different datasets.
        """
        if data_type is 'dummy':
            if override or self.dummy is None:
                file_path = os.path.join(self.path, 'train_example.bson')
                self.dummy = self._get_data(file_path)
            else:
                print("Dummy data is already loaded.")
        elif data_type is 'train':
            raise NotImplementedError
        elif data_type is 'test':
            raise NotImplementedError
        else:
            print("Specified data type \"{}\" not recognized. Please choose from:\
                  [dummy, train, test]".format(data_type))
            
    def _print_sample(self):
        """
        This method prints a random sample from a random dataset that is not
        empty.
        """
        datasets = list([self.dummy, self.train, self.test])
        random.shuffle(datasets)
        for i in datasets:
            if i is not None:
                image = i.sample(n=1)
                plt.title(image['category_id'])
                plt.imshow(image['image'].values[0].reshape((180,180,3)))
                break            
    
    def _get_data(self, file_path):
        """
        This method imports the dummy dataset if it wasn't already loaded.
        """
        data_set = []
        
        print("Reading BSON file...")
        data = bson.decode_file_iter(open(file_path, 'rb'))
        
        print("Starting processing...")
        for c, d in enumerate(data):
            # Store what product belongs to what 
            category_id = d['category_id']
            for e, pic in enumerate(d['imgs']):
                picture = imread(io.BytesIO(pic['picture']))
                # Store image with its label
                data_set.append((category_id, picture.reshape(1, -1)))            
            if c % 10 == 0 and c > 0:
               print("Iteration {}".format(c))
        
        # Convert image data to Pandas DataFrame
        pd_data = pd.DataFrame.from_records(data_set)
        pd_data.rename(columns={0: 'category_id', 1: 'image'}, inplace=True)
        
        return pd_data