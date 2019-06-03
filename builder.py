import data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = data.build_dataset(10000)

# linear regression for card holder risk
inputs = df[['amount', 'holder_index']]
outputs = df['dispute_risk']

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4, random_state=101)

holder_lm = LinearRegression()
holder_lm.fit(X_train, y_train)

predictions = holder_lm.predict(X_test)
