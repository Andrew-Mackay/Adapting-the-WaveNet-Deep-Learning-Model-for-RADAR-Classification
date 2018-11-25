# Set matplotlib backend (much more efficient as no longer showing plot)
import matplotlib
matplotlib.use('Agg')

# coding: utf-8

# In[9]:


WINDOW_LENGTH = 3
# WINDOW_LENGTH = 2
# WINDOW_LENGTH = 1.5
# WINDOW_LENGTH = 1

# In[10]:


import os
if os.getcwd() == '/content':
    from google.colab import drive
    drive.mount('/content/gdrive')
    BASE_PATH = '/content/gdrive/My Drive/Level-4-Project/'
    get_ipython().system('cd gdrive/My\\ Drive/Level-4-Project/ && pip install --editable .')
    os.chdir('gdrive/My Drive/Level-4-Project/')
    
elif os.getcwd() == 'C:\\Users\\macka\\Google Drive\\Level-4-Project\\notebooks\\data_processing' or os.getcwd() == 'C:\\Users\\macka\\Google Drive\\Level-4-Project\\src\\features':
    BASE_PATH = "C:/Users/macka/Google Drive/Level-4-Project/"
    
else:
    BASE_PATH = "/export/home/2192793m/Level-4-Project/"
    
DATA_PATH = BASE_PATH + 'data/'
RAW_PATH = DATA_PATH + 'raw/'
INTERIM_PATH = DATA_PATH + 'interim/'


# In[11]:


from src.features import make_spectrograms, process_labels, make_directory


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import colors
from scipy.signal import butter, freqz, lfilter, spectrogram
import time


# In[13]:


df_labels = pd.read_csv(RAW_PATH + 'Labels.csv')
df_labels.rename(columns={'dataset ID': 'dataset_id'}, inplace=True)


# In[14]:


df_labels = process_labels.process_labels(df_labels)


# In[ ]:


image_width = 150
image_height = 150
minimum_value = 35
norm = colors.Normalize(vmin=minimum_value, vmax=None, clip=True)


# In[32]:


number_of_rows = df_labels.shape[0]
current_row = 1
for row in df_labels.itertuples():
    start_time = time.time()
    print("Processing row", current_row, "of", number_of_rows)
    file_name = RAW_PATH + "Dataset_" + str(row.dataset_id) + ".dat"
    file_path = make_directory.make_directory(
        INTERIM_PATH, WINDOW_LENGTH, row.user_label, row.aspect_angle, row.label)
    
    radar_df = pd.read_table(file_name, sep="\n", header=None)
    spectrograms = make_spectrograms.make_spectrograms(radar_df, WINDOW_LENGTH)
    np.save(file_path + "/" + str(current_row) + "_numpy_spectrogram.npy", spectrograms)  # save matrix version of spectrograms
    count = 1
    for spectrogram in spectrograms:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(image_width, image_height)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(20 * np.log10(abs(spectrogram)), cmap='jet', norm=norm, aspect="auto")
        fig.savefig(file_path + "/" + str(current_row) + "_" + str(count)+".png", dpi=1)
        plt.close(fig)
        count += 1

    current_row += 1
    time_for_row = (time.time() - start_time)/60
    print("---Row took %s minutes ---" % (int(time_for_row)))

