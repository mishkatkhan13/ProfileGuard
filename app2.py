from flask import Flask, render_template , request , abort
import pickle
import requests
import os
import json
import pandas as pd
import numpy as np
import instaloader

from sklearn.tree import DecisionTreeClassifier           # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split      # FOR train_test_split function
from sklearn.metrics import confusion_matrix, accuracy_score
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.errorhandler(404)
def page_not_found(e):
    
    return render_template("404error.html"),404


@app.route('/', methods=['POST'])
def my_form_post():
    global username
    global auth
    username = request.form['username']
    
    def get_profile_info(username):
        L = instaloader.Instaloader()
        try:
            profile = instaloader.Profile.from_username(L.context, username)
            return profile
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"User '{username}' does not exist.")
    
    json_response = get_profile_info(username)

    global comment1

    if request.method == 'POST':
        comment1 = 1       #getting the input from the form
        comment2 = json_response.username
        sentence = json_response.full_name
        words = sentence.split()
        word_lengths = len(words)
        comment3 =  word_lengths
        comment4 = json_response.full_name
        comment5 = 0
        desc = json_response.biography
        words1 = desc.split()
        word_lengthsdesc =len(words1)
        comment6 = word_lengthsdesc
        comment7 = 0
        if(json_response.is_private==True):
            comment8 = 1
        else:
            comment8 = 0
        comment9 = 8
        comment10 = json_response.followees
        comment11 =  json_response.followers

        length = len(comment2)    #finding length of username
        digcount = 0
        for char in comment2:     #finding the number of digits in the username
            if char.isdigit():
                digcount += 1

        ratio = digcount/length    #finding ratio
        import math
        ratio = round(ratio,2)

        namelength = len(comment4)    #length of the account name
        digcount2 = 0
        for char in comment4:         #no of digits
            if char.isdigit():
                digcount2 += 1
        
        ratio2 = digcount2/namelength  #ratio
        ratio2 = round(ratio2,2)


        data1 = comment1
        data2 = ratio
        data3 = comment3
        data4 = ratio2
        data5 = comment5
        data6 = comment6
        data7 = comment7
        data8 = comment8
        data9 = comment9
        data10 = comment10
        data11 = comment11

        import pandas as pd
        import random
        df = pd.read_csv("insta_train.csv")
        df.to_csv("train.csv", header=False, index=False)        #removing heading and index in the dataset
        dataset = pd.read_csv("train.csv")
        X = dataset.iloc[:, 0:10].values                  #storing first 10 columns in x

        Y = dataset.iloc[:, 11].values                    #storing the last column in y
        

        from sklearn.model_selection import train_test_split
        #test_size = 0.2 means only 2% of the whole dataset is sent for testing
        X_train, X_test1, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)  


        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)   #kernal transforms data into linear form
        classifier.fit(X_train, Y_train)                    #fit the svm model according to training data
        X_test = [[data1, data2, data3, data4,data5,data6, data7, data8, data9, data10]]       #giving our input to model
        Y_pred = classifier.predict(X_test)            #store result in Y_pred 
        Y_pred1 = classifier.predict(X_test1)         #predicting for test data

        from sklearn.metrics import confusion_matrix,classification_report
        cm = confusion_matrix(Y_test, Y_pred1)           #comparing test data prediction result with original result
        print(classification_report(Y_test,Y_pred1))        #to calculate precision,recall, F1 score 

        iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)   
        accuracy2=((iclf.score(X_test1, Y_test))*100)   #finding training accuracy

        from sklearn.preprocessing import StandardScaler
        data = pd.read_csv(r"file1.csv")
        y = data['Isfake']
        X = data.drop(columns=['Isfake'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        from sklearn.ensemble import RandomForestClassifier
        
        RF_object = RandomForestClassifier(random_state=42)
        RF_object.fit(X_scaled, y)
        print(len(json_response.username))
        input_features = [
        json_response.followees,
        json_response.followers,
        comment6,
        json_response.mediacount,
        1,
        comment8,
        6,
        len(json_response.username)
        ]
        input_scaled = scaler.transform([input_features])
        prediction = RF_object.predict(input_scaled)[0]
        if prediction == "Yes":
            prediction = "Prediction: Account is most probably fake."
            return render_template('index.html',res='fake',value= json_response.username)
        else:
            prediction = "Prediction: Account is most probably genuine."
            return render_template('index.html',res='real',value= json_response.username)
        # if Y_pred[0] == 1:
        #     response1= 'Fake Profile'
        #     print('Fake Profile')
        #     return render_template('index.html',res='fake',value= json_response.username)
        # else:
        #     response1 = 'Real Profile'
        #     print('Real Profile')
        #     return render_template('index.html',res='real',value= json_response.username)
        

if __name__ == "__main__":
    app.run(debug=True)