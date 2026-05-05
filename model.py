import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Days": [0,30,60,90,120,150,180,210,240,270],
    "Sales": [200,220,250,270,300,320,350,370,390,420]
}

df = pd.DataFrame(data)

X = df[['Days']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame({'Days':[300,330,360,390,420]})
pred = model.predict(future)

plt.plot(df['Days'], y, label="Actual")
plt.plot(future['Days'], pred, label="Predicted")
plt.legend()
plt.title("Sales Prediction")

plt.savefig("prediction.png")
plt.show()
