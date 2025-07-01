#Version 1.0.0
#   De lo que mas enfasis hizo el ingeniero en clases fue aplanar la curva y
#   logre esto gracias a una funcion logaritmica y llegue a 73.03769386136207
#   de acierto al entrenar mi modelo sin sesgarlo
#   Tambien elimine los valores outliers que entorpecian el modelo ya que son
#   Pequeñas exepciones que en lugar de ayudar afectan la pendiente al linealizar
#Version 1.0.1
#   Le agregue mas divisoon de datos y llegue al 74.1510686394193
#   Ademas de que segui jugando con los datos
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Lo de siemore
df = pd.read_csv('housing.csv')

# 1) A tantear el terreno
print(df.head())
print(df['ocean_proximity'].value_counts())
print(df.info())
print(df.describe())

# Histograma general
df.hist(figsize=(15,8), bins=50, edgecolor='black')

# Mapa de calor de correlaciones como en la clase
plt.figure(figsize=(15, 8))
sb.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu')
plt.title('Mapa de Calor de Correlaciones')

# Relación geográfica
sb.scatterplot(data=df, x='longitude', y='latitude', hue='median_house_value', palette='coolwarm')

# 2) Limpieza de datos


df = pd.concat([df, pd.get_dummies(df['ocean_proximity'], dtype=int)], axis=1)
df.drop(['ocean_proximity'], axis=1, inplace=True)
df.dropna(inplace=True)

# 3)Maquillaje de datos



# transformación logarítmica de la variable objetivo

df['median_house_value']= np.log2(df['median_house_value'])

skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
for feature in skewed_features:
    df[feature] = np.log2(df[feature])

# eliminación de valores extremos (outliers) en percentiles 1.9% y 99%
for col in ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']:
    q_low = df[col].quantile(0.019)
    q_high = df[col].quantile(0.99)
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

# 4)  Creacion de mas datos
df['income_per_population'] = df['median_income'] / (df['population'] )
df['rooms_per_household'] = df['total_rooms'] / (df['households'] )
df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'] )
df['income_x_rooms'] = df['median_income'] * df['rooms_per_household']
df['households_per_population'] = df['households'] / (df['population'] )
df['rooms_per_person'] = df['total_rooms'] / (df['population'] )
df['bedrooms_per_person'] = df['total_bedrooms'] / (df['population'] )
df['income_per_room'] = df['median_income'] / (df['total_rooms'] )
df['population_density'] = df['population'] / (df['households'] )


# 5) Ahora si
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)

y_train.shape
modelo = LinearRegression()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

info = {
    'predicciones': predicciones,
    'y_test': y_test
}

pd.DataFrame(info)

print(modelo.score(X_train, y_train))
print(modelo.score(X_test, y_test))

sb.histplot(df['total_rooms'])

sb.scatterplot(df, x='median_house_value', y='median_income')



rmse = mean_squared_error(y_test, predicciones)

print(np.sqrt(rmse))

df.describe()


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

pd.DataFrame(X_test_scaled)

modelo_escalado = LinearRegression()

#entreno el modelo con el conjunto de entrenamiento
modelo_escalado.fit(X_train_scaled, y_train)

predicciones_escaled = modelo_escalado.predict(X_test_scaled)

info_escaled = {
    'predicciones': predicciones_escaled,
    'y_test': y_test
}

pd.DataFrame(info_escaled)

print(modelo_escalado.score(X_train_scaled, y_train)*100)
print("Intento mas alto: 74.1510686394193 %")
