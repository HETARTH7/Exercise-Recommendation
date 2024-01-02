import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('spotify_songs.csv')
dataset.head()

dataset.info()

dataset.shape
