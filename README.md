# Sunflower-Blooming-Stage-Classification
1. This repository contains the source code and raw data for sunflower blooming stage image classification using transfer learning using Inception V3, which itself is one part of a group project carried out by Loh Min Yi, Hneah Guey Ling, Mohammed Jubarah and Chan Khai Shen (me) as the coursework of the Multimodal Information Retrieval subject. I am fully responsible for the Inception V3 part.
2. While the whole project includes a custom Convolutional Neural Network and transfer learning model using VGG-16, AlexNet and Inception V3, this repository only stores the source code for training and evaluating Inception V3, because this is the part that I am responsible for and has access to.
3. The source code for developing, training, experimenting and testing the models is in "Train and experiment model.ipynb", whereas the source code for converting the best performing model to tensorflow.js compatible format is in "Convert model.ipynb".

# Dataset
1. The models are trained using 1076 self-collected images, which is splitted into training dataset (70%), validation dataset (15%) and test dataset (15%).
2. The images are cropped and resized into square images with size of 224 pixels * 224 pixels and has 3 channels (Red, Blue, Green).
3. The dataset classifies sunflower blooming stage into 4 classes, namely bud, partially bloom, fully bloom and wilt.
4. The dataset is not included in this repository.

# Model
1. At the first few layers, the model has 3 random data augmentation layers to reduce overfitting. These layers are random horizontal flipping, random rotation and random contrast. This does not change the dataset size but lets the model sees a slightly different version of the dataset during training.
2. After data augmentation, the dataset passes through the default pre-processing layer for Inception V3, which changes the pixel values from between 0 and 255 to between 0 and 1.
3. After that is the Inception V3 base model, which is pre-trained on the ImageNet dataset. Transfer learning is used here so that the weights trained with large dataset (ImageNet) can be reused for an image classification task with small dataset (the sunflower blooming stage dataset). This reduces the dataset preparation burden, while improving the accuracy of the model.
4. In this project, the fully connected layers of Inception V3, which is configured for 1000 classes in ImageNet is substituted with a custom fully connected neural network with 4 classes, which corresponds to the 4 blooming stages of sunflower in this task.
5. The substituted layers consists of a drop out layer, a flatten layer and three dense layers. The drop out layer randomly switches off 20% of the neurons during training to reduce overfitting. The flatten layer flattens the dataset into 1-dimentional array. The first dense layer has 500 neurons with Rectified Linear Unit (ReLU) activation, whereas the second dense layer has 200 neurons with ReLU activation, whereas the third dense layer has 4 neurons with SoftMax activation.
6. The model which achieved the highest accuracy among all models based on transfer learning of Inception V3 (see details in experimentation section), which is the model trained with Adam optimizer with a learning rate of 0.0001 for 10 epochs, is converted to a tensorflow.js compatible, ready-to-use format. The .json file and .bin files of the model is in "tensorflowjs compatible model (adam unfreeze 10)" folder.

# Experimentation
1. For Inception V3 part (the part that I am responsible for), 2 experiments were conducted. 
2. The first experiment compared the effect of different optimizer on the accuracy of the model. The optimizers tested were Adam, RMS Prop, SGD and Adagrad. All pre-trained convolutional layers were frozen. All 4 models were trained for 10 epochs with a learning rate of 0.0001.
3. In the first experiment, it is found that the model trained with Adam optimizer achieved the highest accuracy (validation accuracy 91.61%, test accuracy 91.67%). The raw data for this experiment is in "histories_inception_V3_Adam_freeze_100.csv", "histories_inception_V3_RSMProp_freeze_100.csv", "histories_inception_V3_SGD_freeze_100.csv", "histories_inception_V3_Adagrad_freeze_100.csv" and "test_histories_10_epochs.csv".
4. The second experiment compared the effect of different percentage of unfrozen pre-trained convolutional layers on the accuracy. The total number of the pre-trained convolutional layers in Inception V3 is 311. The percentage of unfrozen layers tested were 10% (31 layers), 15% (47 layers), 20% (62 layers), 25% (78 layers). All 4 models were trained for 10 epochs with a learning rate of 0.0001 with Adam optimizer because Adam optimizer had the best performance in experiment 1.
5. In the second experiment, it is found that the model trained with 10% pre-trained concolutional layers unfrozen achieved the highest accuracy (validation accuracy 94.19%, test accuracy 96.79%). The raw data for this experiment is in "histories_inception_V3_Adam_unfreeze_10_epoch_10.csv", "histories_inception_V3_Adam_unfreeze_15_epoch_10.csv", "histories_inception_V3_Adam_unfreeze_20_epoch_10.csv", "histories_inception_V3_Adam_unfreeze_25_epoch_10.csv" and
"test_histories_10_epochs.csv"
6. The bar charts of the 2 experiments are in "Bar chart of accuracy vs optimizer (experiment 1).png" and "Bar chart of accuracy vs percentage of unfrozen layers (experiment 2).png".

# User guide
1. Open the source code in "Train and experiment model.ipynb" and "Convert model.ipynb" using Google Colaboratory.
2. Running the source code can replicate the exact same development, training, experimentation and testing process, but may not replicate the exact same result because of the randomness in the training process of machine learning models.
