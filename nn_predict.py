import numpy as np
import json
import tensorflow as tf


# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

# === 以下為訓練與儲存模型架構/權重 ===
def train_and_save():


    # 1. 載入資料
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # 2. 建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
        tf.keras.layers.Dense(128, activation='relu', name="dense1"),
        tf.keras.layers.Dense(64, activation='relu', name="dense2"),
        tf.keras.layers.Dense(10, activation='softmax', name="dense3")
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. 訓練模型
    model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.1)

    # 4. 儲存模型架構（簡化為自定格式，方便 numpy 推論）
    model_arch = []
    for layer in model.layers:
        ltype = type(layer).__name__
        lname = layer.name
        cfg = {}
        if ltype == "Dense":
            cfg["activation"] = layer.activation.__name__
            cfg["units"] = layer.units
        elif ltype == "Flatten":
            pass
        # 權重名稱
        wnames = []
        for i, w in enumerate(layer.get_weights()):
            wnames.append(f"{lname}_w{i}")
        model_arch.append({
            "name": lname,
            "type": ltype,
            "config": cfg,
            "weights": wnames
        })
    with open("model/fashion_mnist.json", "w") as f:
        json.dump(model_arch, f)

    # 5. 儲存權重為 npz
    weights = {}
    for layer in model.layers:
        for idx, w in enumerate(layer.get_weights()):
            weights[f"{layer.name}_w{idx}"] = w
    np.savez("model/fashion_mnist.npz", **weights)

    print("模型架構已儲存為 model/fashion_mnist.json，權重已儲存為 model/fashion_mnist.npz")

# === 主程式 ===
if __name__ == "__main__":
        train_and_save()

