import jax
import jax.numpy as jnp
from flax import serialization
import optax
from flax.training import train_state
import time
import base64
from model import CNN
from utils import load_data, numpy_to_jax


def create_state(rng):
    model = CNN()
    params = model.init(rng, jnp.ones([1, 784]))['params']
    tx = optax.adamw(learning_rate=0.001, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss + 1e-4 * l2_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return state, loss, accuracy


def evaluate(state, X, y):
    logits = state.apply_fn({'params': state.params}, X)
    return jnp.mean(jnp.argmax(logits, -1) == y)


def save_model(state):
    bytes_output = serialization.to_bytes(state.params)
    base64_str = base64.b64encode(bytes_output).decode('utf-8')
    with open('model_base64.txt', 'w') as f:
        f.write(base64_str)
    print("Модель сохранена успешно")


def train():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = numpy_to_jax(X_train), numpy_to_jax(X_test)
    y_train, y_test = numpy_to_jax(y_train), numpy_to_jax(y_test)

    print("Данные загружены, начинаем обучение...")

    state = create_state(jax.random.PRNGKey(42))
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(15):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(X_train))
        for i in range(0, len(X_train), 256):
            batch = X_train[perm[i:i + 256]]
            labels = y_train[perm[i:i + 256]]
            state, loss, acc = train_step(state, batch, labels)

        test_acc = evaluate(state, X_test, y_test)
        train_acc = evaluate(state, X_train[:10000], y_train[:10000])

        print(f"Epoch {epoch + 1:2d} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Test Acc: {test_acc:.3f} | "
              f"Time: {time.time() - start_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(state)
            print(f"Новая лучшая точность: {best_acc:.3f}, модель сохранена")

    print(f"\nЛучшая точность на тесте: {best_acc:.3f}")

    try:
        with open('model_base64.txt', 'r') as f:
            saved_model = f.read()
        print("Проверка сохранённой модели успешна")
    except Exception as e:
        print(f"Ошибка при проверке сохранённой модели: {e}")

    return state


if __name__ == "__main__":
    trained_state = train()