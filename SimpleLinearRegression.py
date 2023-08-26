import numpy as np
import matplotlib.pyplot as plt
# matplotlib for data visualization

plt.style.use('grayscale')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in the 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"x_train.shape: {x_train.shape}")
# or m = len(x_train)
m = x_train.shape[0]
print(f"Number of training examples is {m}")


# seeing the i_th observation:
i = 0
while i < m:
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"x^({i}), y^({i}) = ({x_i}, {y_i})")
    i += 1


# Plot the data, marker = shape, c = color
plt.scatter(x_train, y_train, marker='X', c='k')
plt.title("House prices")
plt.ylabel("Price(in 100s of dollars)")
plt.xlabel("Size(1000 sqft)")
# plt.show()


# regression function: f(x) = wx + b
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = x[i] * w + b
    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b)
# plot the model
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
# plot the data
plt.scatter(x_train, y_train, marker='X', c='k', label='Actual Values')
plt.title('House Prices')
plt.ylabel("Price(in 100s of dollars)")
plt.xlabel("Size(1000 sqft)")
plt.legend()
plt.show()

x_to_predict = 1.5
y_predicted = w * x_to_predict + b


