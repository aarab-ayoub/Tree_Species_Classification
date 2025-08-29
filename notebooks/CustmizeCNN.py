import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import sys
import time


class CustomCNN(nn.Module):
	# Define the CNN architecture 2D convolution 

	def __init__(self, input_channels, num_classes):
		super(CustomCNN, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust based on input size
		self.fc2 = nn.Linear(512, num_classes)
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(x.size(0), -1)  # Flatten the tensor
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

	def count_parameters(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	