# 400_LSTM
Code will generate thin film deposition data. Then, it will train the RNN- LSTM model. After that it will perform the test.
First, open Preprocessing_400.py file. You need to enter the the number of deposition data at he beginning of the file. It will assume you entered 20000 as input if you dont change it. Run the file and wait until the necessary txt file are created. 
Then, open READ_FILES.py file. Run it. After you run it, you can open the MODEL_1.py file and uncomment the model training part if you want to train a new model. I trained the model and save it trained_model9.h5. If you want to train the model, please dont forget to change the name.
In this file, it will automatically evaluate the performance of the model gives the result.
Reminder: Please read the .py files before you start to experimenting on it. I added the command lines which explain the code.
