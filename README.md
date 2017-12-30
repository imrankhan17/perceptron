# Implementing the Perceptron neural network with Numpy

This is a simple Python script which implements the single-layer Perceptron (or linear threshold unit) in Numpy.  It is capable of learning the basic logical operators, namely AND, OR, NOT and any arbitrary combination thereof.

### Usage

```
>>> from perceptron import Perceptron
```

Specify your chosen logical operator and instantiate the `Perceptron` object.
```
>>> network = Perceptron('and')
```

Now pass your inputs as a size 2 array.
```
>>> network.output([False, False])
0.0
```

Equivalently:
```
>>> network.output([0, 0])
0.0
```

Also, supports multiple input arrays.
```
>>> network = Perceptron('or')
>>> network.output([[0, 0], [0, 1], [1, 0], [1, 1])
array([ 0.,  1.,  1.,  1.])
```

Finally, the class supports arbitrary weights and biases with one example being provided.
```
>>> network = Perceptron('ARBITRARY')
>>> network.output([
...     [0, 0, 0], 
...     [0, 0, 1],
...     [0, 1, 0],
...     [1, 0, 0],
...     [0, 1, 1],
...     [1, 1, 0],
...     [1, 0, 1],
...     [1, 1, 1]
... ])
array([ 0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.])
```
