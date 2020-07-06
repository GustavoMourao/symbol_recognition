# ASL interpretation


Based on features get from LeapMotion sendor, using SDK toolit, this project uses Sckit-learn library to interpret American Signal Languages symbols.

Three models were been analyzed:

  - ANN (Artificial neural netwod)
  - KNN
  - SVM
  - Nearest Centroid

## Features and training

The set of features used to train the set of models were (get from LeapMotion sensor SDK): 
  - RTP
  - RTT
  - RTJ

To get trained model:
  - Run script: 
```sh
python train.py --classifierType ANN
```
to get Confusion Matrix from SVM, for example.

The list of available models can be visualized with:
```sh
python train.py --help
```

## Running trained model

All the trained models are in the directory ```data/```.
To run the ANN trained model, type:
```sh
python app.py
```

  - Read the document: Letter_Leap.pdf to get more information


In case of you want the real time application, please send me a message!

![Alt Text](https://github.com/GustavoMourao/symbol_recognition/tree/master/data/asl-gmourao.gif)