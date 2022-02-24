# Import Modules

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Data

iris = load_iris()

# Split Data

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Create model
model = LogisticRegression(max_iter=5000)
fit = model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# Model is 93.33% Accurate