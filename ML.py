#!/usr/bin/env python
# coding: utf-8
# Import Block

try:
    import os
except ImportError:
    raise ImportError(" 'os' package not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV package not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import numpy as np
except ImportError:
    raise ImportError("Numpy not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import sklearn
    from sklearn.metrics import classification_report
except ImportError:
    raise ImportError("Sklearn not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(" Matplotlib.pyplot not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
except ImportError:
    raise ImportError(" IPyWidgets not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------#    

try:
    import logging
except ImportError:
    raise ImportError("Logging module not found. Check Chemistream Installation") 
#-------------------------------------------------------------------------------------#   

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import datasets
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical
except ImportError: 
    raise ImportError("Tensorflow not found. Check Chemistream Installation")
#-------------------------------------------------------------------------------------# 

class MLexample:
    
    def __init__(self):
        """ Constructor
        Class variables : x_train, x_test, y_train, y_test, model, class_names"""
        pass
    
    def output_shapes(self):
        
        """Simple Print Method - Once the dataset is selected this method is
        used to print the shapes of Train and Test sets
        
        Args:
            None 
        
        Returns: 
            Shapes of matrices"""
        
        # Shapes of Train Dataset
        print("\nTraining Set -",end="\t")
        print("x_train.shape {} &".format(self.x_train.shape),end="\t")
        print("y_train.shape {}".format(self.y_train.shape))
        
        # Shapes of Test Dataset
        print("Test Set -",end="\t")
        print("x_test.shape {} &".format(self.x_test.shape),end="\t")
        print("y_test.shape {}".format(self.y_test.shape))
    
        
    def set_data(self):
        
        """Widget - Dropdown Box to select the dataset. Allows user to select from available MNIST datasets:
        - Digits_Mnist
        - Fashion_Mnist
        
        Args: 
            None
        
        Returns: 
            dataSet: Name of selected dataset"""
        
        # This widget calls the get_data method
        interact(self.get_data,
                 dataSet = widgets.Dropdown(options=['Digits_Mnist','Fashion_Mnist'],description='Dataset:'))
    
    def get_data(self, dataSet):
        
        """Allows user to select the dataset of their choice and then divides the dataset 
        into Train set i.e. x_train, y_train and Test set x_test, y_test.
        Where x is the matrix with each image's pixel intensity values (0-255) laid out as row vectors
        and y is the matirx with integer values of true labels corresponding to class_names's list indices.
        
        Args: 
            dataSet : Name of selected dataset
        
        Returns :
            None"""
        
        print("\nDataset Selected: {}".format(dataSet))
        
        if dataSet == 'Digits_Mnist':
            
            # Downloading Data 
            digits_mnist = datasets.mnist 
            
            # Splitting Dataset
            (self.x_train, self.y_train), (self.x_test, self.y_test) = digits_mnist.load_data()
            
            # Normalizing Pixel Intensities
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0 
            
            # Assigning Class Names
            self.class_names = ['ZERO','ONE', 'TWO', 'THREE', 'FOUR', 'FIVE','SIX', 'SEVEN', 'EIGHT', 'NINE']
            print("Classes:",self.class_names)
            
            # Outputs the shapes of Train and Test datasets
            self.output_shapes() 
            
        elif dataSet == 'Fashion_Mnist':
            
            # Downloading Data
            fashion_mnist = datasets.fashion_mnist 
            
            # Splitting Dataset
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
            
            # Normalizing Pixel Intensities
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0 
            
            # Assigning Class Names
            self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt','Sneaker','Bag','Ankle boot']
            print("Classes:",self.class_names)
            
            # Outputs the shapes of Train and Test datasets
            self.output_shapes() 
        
        else:
            pass       
        
        
        
    def set_viz(self): 
        
        """Widget - Integer Slider to set the number of Images to display.
        Maximum number of images that can be displayed is set to 25 for simple and clear output.
        
        Args: 
            None
        
        Returns:
            number: Selected number of images to display """
        
        print('Slide to select number of Images to display')
        
        # This widget calls the get_viz method
        interact(self.get_viz,
                 number = widgets.IntSlider(min=1,max=25,description="Images:",value=5)) 
    
    def get_viz(self,number):
        
        """Visulazing the dataset. After selecting 'n' number of Images
        to be displyed, first 'n' Images of the dataset are plotted 
        
        Args: 
            number: Number of selected Images to plot
        
        Returns: 
            Plots""" 
        
        plt.figure(figsize=(10,10))
        for i in range(number):
            
            # Plotting 5 Images per row for a cleaner display
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.y_train[i]])
        plt.show()
    
    def load_PTmodel(self, fileName=""):
        
        """Load a pre-trained model from 'FileName.h5'.
        This method can only be used if there is a saved model(.h5) available 
        in the same directory as the Load_PreTrained_Model.ipynb. 
        
        Args: 
            fileName : Name of the saved .h5 file. 
        
        Returns : 
            None"""
        
        # Loads a model from specified filename. 
        self.model = load_model(fileName)    
        self.model.summary()
        print("Successfully loaded the model form {} file".format(fileName))
    
    def set_arch(self):
        
        """ Widget        - Dropdown Box to select architecture and Float Slider to set Dropout value 
            Architectures - 1. 784-128-128-10 2. 784-128-10
            Dropout value - 0 < d <= 0.5
        The architecture needs to be specified before setting dropout value. The get_arch method
        waits for execution untill Drop Out value is set.
        
        Args: None
        
        Returns: 
            arch    : Selected Architecture
            dropOut : Dropout value selected"""
        
        print("Select the Architecture for the model and then Set Dropout value (0 < d <= 0.5)")
        
        # This widget calls the get_arch method
        interact(self.get_arch, 
                 dropOut= widgets.FloatSlider(min=0,max=0.5,step=0.1,value='0',description="Dropout:"),
                 arch=widgets.Dropdown(options=['784-128-128-10', '784-128-10'],description='Architecture:')) 
                                         
        
    def get_arch(self,arch,dropOut):
        
        """Selected Architecture of the model is constructed with selected value for dropout layers
        
        Args: 
            arch    : Selected Architecture
            dropOut : Dropout value selected
        
        Returns: 
            None
        """   
        
        # To avoid execution before assignment
        if dropOut == 0:
            pass
        
        else:
            
            print("Loading the architecture: {}".format(arch))
            print("With Dropout Value set to: {}".format(dropOut))
        
            if arch == '784-128-128-10':
                
                # Keras model description
                self.model = tf.keras.models.Sequential(name='Sequential_1')
                self.model.add(Flatten(input_shape=(28, 28),name='flatten_1')),
                self.model.add(Dense(128, activation='relu',name='dense_1')),
                self.model.add(Dropout(dropOut,name='dropOut_1')),
                self.model.add(Dense(128, activation='relu',name='dense_2')),
                self.model.add(Dropout(dropOut,name='dropOut_2')),
                self.model.add(Dense(10, activation='softmax',name='dense_3'))
                
            elif arch == '784-128-10':
                
                # Keras model description
                self.model = tf.keras.models.Sequential(name='Sequential_1')
                self.model.add(Flatten(input_shape=(28, 28),name='flatten_1')),
                self.model.add(Dense(128, activation='relu',name='dense_1')),
                self.model.add(Dropout(dropOut,name='dropOut_1')),
                self.model.add(Dense(10, activation='softmax',name='dense_2'))
                
            # Model Summary
            print("\nModel Summary")
            self.model.summary()            

    
    def set_HyperParameters(self):
        
        """Widget    - Set Optimizer, Loss function and choose Metrics to set model Hyper Parameters. 
           Optimizer - 1. Adam 2. SGD - Stochastic Gradient Descent
           Losses    - 1. Poisson 2. Sparse Categorical Cross Entropy
           Metrics   - 1. Accuracy 2. MSE - Mean Squared Error
           
           Args: None
           
           Returns:
               Opt     : Selected Optimizer
               Loss_val: Selected Loss Measure
               metric  : Selected Metrics"""

        print("Select the Optimizer, Loss Function and performance Metrics")
        
        # This widget calls the compile_model method
        interact(self.compile_model, 
                 Opt=widgets.Dropdown(options=['Select Optimizer', 'Adam','SGD'],description="Optimizer:"),
                 Loss_val=widgets.Dropdown(options=['Select Loss Measure','poisson','sparse_categorical_crossentropy'],description="Loss Measure:"),
                 Mets=widgets.Dropdown(options=['Select Metrics','accuracy','mse', 'Both'],description="Metrics:"))
    
    
    def compile_model(self,Opt,Loss_val,Mets):
        
        """Compiling the model using, user selected values for HyperParameters 
        from set_HyperParameters method
        
        Args:
            Opt     : Selected Optimizer
            Loss_val: Selected Loss Measure
            metric  : Selected Metrics
        
        Returns: 
            None"""        
        
        # To avoid execution before assignment
        if Opt == 'Select Optimizer' or Loss_val == 'Select Loss Measure' or Mets == 'Select Metrics':
            pass
       
        else :

            if Mets == 'Both':
                metric = ['accuracy','mse']
            else: 
                metric = [Mets]

            print("Optimizer set to -> {}".format(Opt))
            print("Loss Measure set to -> {}".format(Loss_val))
            print("Metrics selected are -> {}".format(metric))
            
            # Complie Model
            self.model.compile(optimizer=Opt,loss=Loss_val,metrics=metric)
            
    def set_epochs(self):
        
        """Widget - Int Slider to input the number of Epochs for Training
        
        Args: 
            None
        
        Returns:
            epoch: Selected number of training iterations"""
        
        print("Select number of Epochs for Training")
        
        # This widget calls the train_model method
        interact(self.train_model, 
                 epoch=widgets.Dropdown(options=['0','1','5','10','15'],description='Epochs:'))        
        
    def train_model(self,epoch=''):
        
        """Train the model on training dataset
        
        Args: 
            epoch : Selected number of training iterations
        
        Returns: 
            None"""
        
        # To avoid execution before assignment
        if int(epoch) == 0:
            pass
        else:
            self.model.fit(self.x_train, self.y_train, epochs= int(epoch))
        
    def evaluate_model(self):
        
        """Evaluate the model's performance on Test dataset
        
        Args: 
            None
        
        Returns: 
            Model evaluation statistics"""
        
        ev = self.model.evaluate(self.x_test, self.y_test,verbose=0)
        print("\nTest Accuracy = {0:.2f}%".format(ev[1]*100))
        print("Test Loss = {0:.4f}".format(ev[0]))
        predictions = self.model.predict(self.x_test)
        
        preds = []
        for p in predictions:
            a = np.argmax(p)
            preds.append(a)
        print("Classification Report\n",classification_report(preds,self.y_test))
        
    
    def save_model(self,name=''):
        
        """Saves the model (Architectures and Weights together) with the specified name
        
        Args: 
            name: Given name to save file as
        
        Returns: 
            Saved file details"""
        
        name = name + '.h5'
        self.model.save(name)
        print("Model saved as {}".format(name))
      
    def set_plots(self):
        
        """Widget - Int Slider to browse through predictions of test set images
        
        Args:
            None
        
        Returns: 
            image: Index of the Image to be displayed"""
        
        style = {'description_width':'initial'}
        
        # This widget calls the plot_preds method
        interact(self.plot_preds,image=widgets.IntSlider(min=1,max=10000,
                                    description='Test Image', style=style))   

    def plot_preds(self,image):
        
        """Function to plot predictions of Test images
        
        Args:
            image: Index of the Image to be displayed
        
        Returns:
            Plots"""
        
        plt.imshow(self.x_test[image-1],cmap=plt.cm.binary)
        predictions = self.model.predict(self.x_test)
        predictName = self.class_names[(np.argmax(predictions[image-1]))]
        testName    = self.class_names[self.y_test[image-1]]
            
        plt.title("Prediction: " + predictName + " --> " + "Label: " + testName)
        plt.pause(0.05)
        
   #--------------------------------------------------------------------------------------------------------------# 
    def prep_data(self,path=""):
        
        """Method to read in new image data from the specified location(Path) to make predictions.
        
        Args:
            path: The path for Validation set of images
            
        Retuns:
            Plots the new image data to visualize"""

        images = []
        X = []
        r = os.walk(path)
        for roots, dirs, files in r:
            for j in files:
                if j == '.DS_Store':
                    pass
                else:
                    p = path + "/" + j
                    img = cv2.imread(p,0)
                    img = cv2.resize(img,(28,28))
                    X.append(img)
                    
        # Normalizing Pixel Intensity Values            
        self.X = (np.asarray(X))/255.0 
        
        plt.figure(figsize=(10,5))
        for i in range(len(X)):
            plt.subplot(2,5,i+1)            
            plt.imshow(X[i],cmap=plt.cm.binary)
        
        print("Preprocessed the image data into matrix of the shape {}".format(self.X.shape))
        print("Number of Images available for predictions is: {}".format(len(X)))
    
    def pred_and_plot(self,index, dataType):
        
        """ Make predictions on new data and plot images and their labels.
        
        Args:
            index: The index of the image to visualize.
            dataType: Specifying the image data type to get corresponding label names. 
        
        Returns:
            Visualizations of images and their respective predictions. 
        """

        preds = self.model.predict(self.X)
        predictions = []
        for pre in preds:
            predictions.append(np.argmax(pre))
        
        if dataType == "Digits":
            class_names = ['ZERO','ONE', 'TWO', 'THREE', 'FOUR', 'FIVE','SIX', 'SEVEN', 'EIGHT', 'NINE']
            plt.imshow(self.X[index],cmap=plt.cm.binary)
            print("Prediction: ",class_names[predictions[index]])
        
        elif dataType == "Fashion":
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt','Sneaker','Bag','Ankle boot']
            plt.imshow(self.X[index],cmap=plt.cm.binary)
            print("Prediction: ",class_names[predictions[index]])        
        
        else:
            print("Sorry no labels available for this dataset")
            plt.imshow(self.X[index],cmap=plt.cm.binary)
            print("Predictions of label (index of 0-9)".format([predictions[index]]))