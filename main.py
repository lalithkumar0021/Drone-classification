import pandas as pd
import tkinter as tk
from tkinter import*
from tkinter import filedialog

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
import pickle

import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
main = tk.Tk()
main.title(" MACHINE LEARNING ANALYSIS OF DRONE CLASSIFICATION MODELS: INSPIRE, MAVIC, PHANTOM, AND NO DRONE")
main.geometry("1600x1300")
title = tk.Label(main, text="MACHINE LEARNING ANALYSIS OF DRONE CLASSIFICATION MODELS: INSPIRE, MAVIC, PHANTOM, AND NO DRONE",justify='center')

model_folder = 'model'
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r"Dataset"
model_folder = 'model'

#create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
flat_data_file = os.path.join(datadir, 'flat_data.npy')
target_file = os.path.join(datadir, 'target.npy')

def upload():
    global filename
    global dataset,categories
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    path = r"Dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    categories
    text.insert(END,"Total Categories Found In Dataset"+str(categories)+'\n\n')

Categories=['dji_inspire', 'dji_mavic', 'dji_phantom', 'no_drone']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r"Dataset"
model_folder = 'model'

def imageprocessing():
    global flat_data,target,categories
    
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir=r"Dataset"
    #create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')

    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        text.insert(END,"Total Images Found In Dataset : "+str(flat_data.shape[0])+'\n\n')
        
    else:
        #path which contains all the categories of images
        for i in Categories:
        
            print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            #create file paths by combining the datadir (data directory) with the i
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))#Reads the image using imread.
                img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
                flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
                target_arr.append(Categories.index(i)) #Adds the index of the category to the target_arr.
                    #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
                print(f'loaded category:{i} successfully')
                #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
        
        
def splitting():
    global x_train,x_test,y_train,y_test
    
    df=pd.DataFrame(flat_data)
    df['Target']=target #associated the numerical representation of the category (index) with the actual image data
    
    x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.20,random_state=77)
    text.insert(END,"Total Images Used For Training : "+str(x_train.shape[0])+'\n\n')
    text.insert(END,"Total Images Used For Testing : "+str(x_test.shape[0])+'\n\n')


labels=Categories
precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+' Accuracy    : '+str(a)+'\n')
    text.insert(END,algorithm+' Precision   : '+str(p)+'\n')
    text.insert(END,algorithm+' Recall      : '+str(r)+'\n')
    text.insert(END,algorithm+' FSCORE      : '+str(f)+'\n')
    report=classification_report(predict, testY,target_names=labels)
    conf_matrix = confusion_matrix(testY, predict)
    text.insert(END,algorithm+" Accuracy : "+str(a)+'\n\n')
    text.insert(END,algorithm+"Classification Report: "+'\n'+str(report)+'\n\n')
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def SVM():
    
    if os.path.exists('SVM_model.pkl'):
        # Load the trained model from the file
        clf = joblib.load('SVM_model.pkl')
        print("Model loaded successfully.")
        predict = clf.predict(x_test)
        calculateMetrics("Support Vector Machine Classifier", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        clf = SVC(C=2,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=False,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,)
        clf.fit(x_train, y_train)
        # Save the trained model to a file
        joblib.dump(clf, 'SVM_model.pkl')
        print("Model saved successfully.")
    predict = clf.predict(x_test)
    calculateMetrics("Support Vector Machineclassifier", predict, y_test)    
    
def RFC():
    global classifier
    text.delete('1.0', END)
    # Check if the pkl file exists
    Model_file = os.path.join(model_folder, "RF_Classifier.pkl")
    if os.path.exists(Model_file):
        # Load the model from the pkl file
        rf_classifier = joblib.load(Model_file)
        predict = rf_classifier.predict(x_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
    else:
        # Create Random Forest Classifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(rf_classifier, Model_file)  
        predict = rf_classifier.predict(x_test)
		#print("Random Forest model trained and model weights saved.")
        calculateMetrics("RandomForestClassifier", predict, y_test)
        report = classification_report(y_test, predict)
        
def prediction():
    global rf_classifier,Model_file
    path = filedialog.askopenfilename(initialdir = "testing")
    img=imread(path)
    img_resize=resize(img,(150,150,3))
    img_preprocessed=[img_resize.flatten()]
    Model_file = os.path.join(model_folder, "RF_Classifier.pkl")
    rf_classifier = joblib.load(Model_file)
    output_number=rf_classifier.predict(img_preprocessed)[0]
    output_name=categories[output_number]

    plt.imshow(img)
    plt.text(10, 10, f'Predicted Output: {output_name}', color='white',fontsize=12,weight='bold',backgroundcolor='black')
    plt.axis('off')
    plt.show()
    
   
title.grid(column=0, row=0)
font=('times', 13, 'bold')
title.config(bg='purple', fg='white')
title.config(font=font)
title.config(height=3,width=120)
title.place(x=60,y=5)

uploadButton = Button(main, text="Upload Dataset   ",command=upload)
uploadButton.config(bg='Skyblue', fg='Black')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Image Processing ",command=imageprocessing)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=250,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Splitting   ",command=splitting)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=450,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="SVM_classifier",command=SVM)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=600,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="RFC  Classifier ",command=RFC)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=770,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Prediction   ",command=prediction)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=950,y=100)
uploadButton.config(font=font)

font1 = ('times', 12, 'bold')
text=Text(main,height=28,width=180)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=15,y=250)
text.config(font=font1)
main.mainloop()
