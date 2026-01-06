# Building-n-layers-neural-network-from-scartch

🧠 N-Layer Neural Network from Scratch (NumPy)

This repository contains a fully vectorized implementation of an N-layer (deep) neural network built from scratch using NumPy only, without relying on deep learning frameworks such as TensorFlow, PyTorch, or Keras.

The code is beginner friendly so every one can understand and benifit from it, only you need is a math

The project generalizes a 2-layer neural network to arbitrary depth, demonstrating a clear understanding of forward propagation, backpropagation, and gradient descent across multiple layers.

📌 Model Architecture (Generalized)
Input Layer
    ↓
Hidden Layer 1 (ReLU)
    ↓
Hidden Layer 2 (ReLU)
    ↓
...
    ↓
Hidden Layer L−1 (ReLU)
    ↓
Output Layer (Softmax)


Example configuration:

layer_dims = [784, 64, 32, 10]


Which corresponds to:

784 → 64 → 32 → 10

🧮 Mathematical Formulation

For an N-layer network:

Forward Propagation (Layer l)
𝑍
(
𝑙
)
=
𝑊
(
𝑙
)
𝐴
(
𝑙
−
1
)
+
𝑏
(
𝑙
)
Z
(l)
=W
(l)
A
(l−1)
+b
(l)
𝐴
(
𝑙
)
=
{
ReLU
(
𝑍
(
𝑙
)
)
	
𝑙
<
𝐿


Softmax
(
𝑍
(
𝑙
)
)
	
𝑙
=
𝐿
A
(l)
={
ReLU(Z
(l)
)
Softmax(Z
(l)
)
	​

l<L
l=L
	​

Loss Function

Categorical Cross-Entropy (Log Loss)

𝐿
=
−
1
𝑚
∑
𝑖
=
1
𝑚
∑
𝑐
=
1
𝐶
𝑦
𝑐
(
𝑖
)
log
⁡
(
𝑦
^
𝑐
(
𝑖
)
)
L=−
m
1
	​

i=1
∑
m
	​

c=1
∑
C
	​

y
c
(i)
	​

log(
y
^
	​

c
(i)
	​

)
Backpropagation (General Case)

Starting from the output layer:

𝑑
𝑍
(
𝐿
)
=
𝐴
(
𝐿
)
−
𝑌
dZ
(L)
=A
(L)
−Y

For each layer 
𝑙
=
𝐿
,
𝐿
−
1
,
.
.
.
,
1
l=L,L−1,...,1:

𝑑
𝑊
(
𝑙
)
=
1
𝑚
𝑑
𝑍
(
𝑙
)
𝐴
(
𝑙
−
1
)
𝑇
dW
(l)
=
m
1
	​

dZ
(l)
A
(l−1)T
𝑑
𝑏
(
𝑙
)
=
1
𝑚
∑
𝑑
𝑍
(
𝑙
)
db
(l)
=
m
1
	​

∑dZ
(l)
𝑑
𝑍
(
𝑙
−
1
)
=
𝑊
(
𝑙
)
𝑇
𝑑
𝑍
(
𝑙
)
⊙
ReLU
′
(
𝑍
(
𝑙
−
1
)
)
dZ
(l−1)
=W
(l)T
dZ
(l)
⊙ReLU
′
(Z
(l−1)
)
🛠️ Implementation Details

Language: Python

Libraries: NumPy only

No ML/DL frameworks used

Fully vectorized operations

Parameters stored using Python lists for scalability

Clean separation of:

initialization

forward propagation

backward propagation

parameter updates

📂 Code Structure
.
├── init_params(layer_dims)     # Initialize weights & biases for N layers
├── forward_prop(X, Ws, bs)     # Forward propagation through all layers
├── backward_prop(Zs, As, Ws)   # Backpropagation across N layers
├── update_params(Ws, bs)       # Gradient descent update
├── one_hot(Y)                  # Label encoding
└── training loop

🚀 Training Loop (High Level)
Ws, bs = init_params(layer_dims)

for epoch in range(epochs):
    Zs, As = forward_prop(X_train, Ws, bs)
    dWs, dbs = backward_prop(Zs, As, Ws, X_train, Y_train)
    Ws, bs = update_params(Ws, bs, dWs, dbs, learning_rate)

🎯 Key Learning Outcomes

How deep networks generalize shallow networks

Why lists + loops scale better than hard-coded layers

How gradient flow works across many layers

Why non-linearity is essential for depth

How vectorized backpropagation is implemented for arbitrary depth

⚠️ Notes

This is an educational implementation, not production-optimized

Initialization uses simple random values (no Xavier / He)

No regularization or batch normalization included

Designed for clarity over performance

📌 Why This Project?

Most tutorials stop at 1–2 layers or hide complexity behind frameworks.
This project demonstrates true understanding of deep learning mechanics by scaling neural networks to any number of layers using only linear algebra.

🔜 Possible Extensions

Xavier / He initialization

Dropout

Batch normalization

Accuracy & loss tracking

Modular activation functions

👤 Author

Abraar
GitHub: Abraar77
