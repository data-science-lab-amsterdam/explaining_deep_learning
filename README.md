# Repository which is an extension on models we created for the big data expo 2018. 
## Crucial to have a model which can be created through our other repo: <LINK>
In this repo we create Lime images which aim to cexplain why a deep learning model comes to certrain prediction.

In this repo there are 2 functions:
- label_image.py which creates a prediction (i.e. ss someone wearing glasses)
- create_image.py which creates a Lime image. Should be used as python create_image.py images/ruben.JPG glasses/retrained.pb glasses/labels.txt

In the notebook explaining_rating.ipynb, we step by step show how to create a Lime image. 

Requirements can be found in the requirements.txt
