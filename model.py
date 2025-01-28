from flax import linen as nn

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, 28, 28, 1)
        x = nn.Conv(32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

