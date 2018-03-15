# ASL interpretation

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

Based on features get from LeapMotion sendor, using SDK toolit, this project uses Sckit-learn library to interpret American Signal Languages symbols.

Three models were been analyzed:

  - ANN (Artificial neural netwod)
  - KNN
  - SVM
  - Nearest Centroid

# Features

The set of features used to train the set of models were (get from LeapMotion sensor SDK): 
  - RTP
  - RTT
  - RTJ

You can also:
  - Run script: 
 ```sh
python main_ObtainModels.py
```
to get Confusion Matrix to each models analyzed :)

  - Read the document: Letter_Leap.pdf to get more information


In case of you want the real time application, please send me a message!