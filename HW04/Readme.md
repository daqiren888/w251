# Homework 4: DL 101

### Final AUC 0.92

### Final Model
MyBombasticMLP( \
  (fc1): Linear(in_features=100000, out_features=192, bias=True) \
  (fc2): Linear(in_features=192, out_features=64, bias=True) \
  (fc3): Linear(in_features=64, out_features=1, bias=True) \
  (m): Sigmoid() \
  (droput): Dropout(p=0.2, inplace=False) \
)

### Final Parameters
batch_size = 512 \
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001) \
criterion = nn.BCELoss() \
#criterion = nn.CrossEntropyLoss() \
#criterion = nn.Binary_CrossEntropyWithLogits \
device = 'cuda' \
model = model.to(device) 


## HW04 Requirement

You will run this homework in colab, and build your own MLP architecture.
Provided with the homework is a dataset and the pipeline to process the data and load to GPU. Please make sure you are comfortable with the preprocessing of the data.
We also provide a logistic regression model implemented in pytorch, along with a benchmarked AUC score.

Your task will be to build an Multi layer perceptron, or MLP, and improve on the score achieved by the LR model.
Try things like,

Adding multiple hidden layers (you can reference this prize winning architecture for an initial set of layer dimensions).
As you introduce more parameters, you will probaby need to drop the learning rate to avoid overfitting.
Does dropout help to avoid overfitting.
Add a relu activation between hidden layers.
Experiment with increasing or decreasing batch size. Or a good way to regularise is starting with small batchsizes and increasing batchsize each epoch.
Add a small weight decay to your Adam optimiser.
After you are happy with the results, download the notebook as a html and submit it to ISVC, together with the highest AUC score you achieved.
You can get started here.
