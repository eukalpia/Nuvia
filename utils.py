import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values.astype('float32') / 255.0
    y = mnist.target.astype('int32')
    return train_test_split(X, y, test_size=0.2, random_state=42)

def numpy_to_jax(data):
    return jnp.array(data)