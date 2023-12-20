# PiGCL

The official source code for A New Mechanism for Eliminating Implicit Conflict in Graph Contrastive   

Part of code is referenced from [Deep Graph Contrastive Representation Learning](https://github.com/CRIPAC-DIG/GRACE)

## Environment Setup

- torch 1.13.0
- torch-geometric 2.3.0
- scikit-learn 1.2.1
- numpy 1.23.5

You can install the dependencies using the following command:

```python
pip install -r requirements.txt
```



### Usage

You can follow these commands to train and test a model with a specific dataset:

```python
./run_script_Cora.sh

./run_script_CiteSeer.sh

./run_script_PubMed.sh

./run_script_CS.sh

./run_script_Photo.sh

./run_script_Computers.sh
```



