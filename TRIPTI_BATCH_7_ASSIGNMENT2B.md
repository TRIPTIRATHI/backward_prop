```python

import numpy as np

########### defining inputs ###########
X =  np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
Y =  np.array([1,1,0]).reshape(3,-1)
wh = np.random.uniform(-1,1,12).reshape(4,3)
print(wh)
bh = np.random.uniform(-1,1,3).reshape(1,3)
print(bh)
wout = np.random.uniform(-1,1,3).reshape(3,1)
print(wout)
bout = np.random.uniform(-1,1,1).reshape(1,1)
print(bout)


```

![input_wh_bh](https://github.com/TRIPTIRATHI/backward_prop/blob/master/random_wh_bh.PNG)

```python

############ predicting output & error ########

def relu(X):
  for row in range(len(X)):
    for col in range(len(X[0])):
      if X[row,col] > 0:
        X[row,col] = X[row,col]
      else:
        X[row,col]=0
    
  return X

hidden_layer_input = np.dot(X,wh) + bh
print(hidden_layer_input)
     
hiddenlayer_activation = relu(hidden_layer_input)
print(hiddenlayer_activation)

output_layer_input = hiddenlayer_activation.dot(wout) + bout
output = relu(output_layer_input)
print(output)

E = Y - output
print(E)

```

![output](https://github.com/TRIPTIRATHI/backward_prop/blob/master/output_error.PNG)

```python

###### calculating Slope ##############

def derivative_relu(X):
  for row in range(len(X)):
    for col in range(len(X[0])):
      if X[row,col] > 0:
         X[row,col] = 1
      else:
         X[row,col]=0
  return X

slope_output_layer = derivative_relu(output)
slope_hidden_layer = derivative_relu(hiddenlayer_activation)
print(slope_output_layer)
print(slope_hidden_layer)

```

![slope](https://github.com/TRIPTIRATHI/backward_prop/blob/master/slope.PNG)

```python

######## calculating delta ########
lr = 0.1
d_output=E*slope_output_layer*lr
Error_at_hidden_layer = d_output.dot(np.transpose(wout))
print(Error_at_hidden_layer)
d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

###### updating weights ########
hiddenlayer_activation = np.transpose(hiddenlayer_activation)
wout = wout + hiddenlayer_activation.dot(d_output) * lr
print(wout)
X_trans = np.transpose(X)
wh = wh + np.dot(X_trans,d_hiddenlayer) * lr
print(wh)

######## updating biasedness #######
bh = bh + np.sum(d_hiddenlayer, axis=0) * lr
bout = bout + np.sum(d_output, axis=0)*lr
print(bh)
print(bout)

```

![updated_wh_bh](https://github.com/TRIPTIRATHI/backward_prop/blob/master/updated_wh_bh.PNG)
