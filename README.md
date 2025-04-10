# Neural-net Extracellular Trained Flux Balance Analysis, a hybrid approach to constrain genome-scale models

## Summary
NEXT-FBA is a hybrid mechanistic and data-driven model to constrain genome scale models (GEMs).
NEXT-FBA involves training an artificial neural network (ANN) to understand the correlation between exometabolomics and cell metabolism. The ANN predicts intracellular reaction bounds for unseen data and these bounds are used to constrain a GEM. This protocol will walk through two separate procedures. Firstly, how to train the ANN on a new dataset. Secondly, applying a pre-trained ANN to constrain a Chinese hamster ovary (CHO) cell GEM.

For complete details on the use and execution of this protocol, please refer to "NEXT-FBA Protocol.pdf" and the NEXT-FBA publication (https://doi.org/10.1016/j.ymben.2025.03.010). Note: This method works with TensorFlow Versions 2.10 and earlier.

## Graphical abstract

![image](https://github.com/J-Morrissey/NEXT-FBA/assets/109590884/1cfb49c1-c7f4-4c8e-af28-06fe6bf49008)

