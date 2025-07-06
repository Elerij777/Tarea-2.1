#Version 1.0.0
#   De lo que mas enfasis hizo el ingeniero en clases fue aplanar la curva y
#   logre esto gracias a una funcion logaritmica y llegue a 73.03769386136207
#   de acierto al entrenar mi modelo sin sesgarlo
#   Tambien elimine los valores outliers que entorpecian el modelo ya que son
#   Pequeñas exepciones que en lugar de ayudar afectan la pendiente al linealizar
#Version 1.0.1
#   Le agregue mas division de datos y llegue al 74.1510686394193
#   Ademas de que segui jugando con los datos
#Version 1.0.2
#   Tal vez hice trampa pero al usar RandomForestRegressor llegue al 83.564284500018%
#   ya que es perfecto para datos con outliers muy grandes como este ejemplo
#   Pero trate los datos y con regresion lineal llegue al 74.21671362426237 %
#Version 1.1.0
#   segui jugando con las variables e hice un clustering geografico y no solo multiplique tambien eleve al cuadrado
#   y cree mas relaciobes y llegue al 77.21162632708844%
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV

# Lo de siempre
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
dummies = pd.get_dummies(df['ocean_proximity'], dtype=int)  # Sin drop_first
df = pd.concat([df , dummies], axis=1)
df.drop(['ocean_proximity'], axis=1, inplace=True)
df.dropna(inplace=True)

# 3) Maquillaje de datos

# transformación logarítmica de la variable objetivo
df['median_house_value'] = np.log2(df['median_house_value']+1)

# transformación logarítmica de features sesgadas
skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
for feature in skewed_features:
    df[f'logaritmo_{feature}'] = np.log2(df[feature] + 1)

# eliminación de valores extremos (outliers) en percentiles 1.9% y 99%
for col in ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']:
    q_low = df[col].quantile(0.019)
    q_high = df[col].quantile(0.99)
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

# 4) Creacion de más datos
df['income_per_population'] = df['median_income'] / (df['population'] )
df['rooms_per_household'] = df['total_rooms'] / (df['households'] )
df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'])
df['income_x_rooms'] = df['median_income'] * df['rooms_per_household']
df['households_per_population'] = df['households'] / (df['population'])
df['rooms_per_person'] = df['total_rooms'] / (df['population'])
df['bedrooms_per_person'] = df['total_bedrooms'] / (df['population'])
df['income_per_room'] = df['median_income'] / (df['total_rooms'] )
df['population_density'] = df['population'] / (df['households'] )

df['population_per_household'] = df['population'] / df['households']
df['households_per_person'] = df['households'] / df['population']
df['rooms_density'] = df['total_rooms'] / (df['longitude']**2 + df['latitude']**2)**0.5
df['median_income_squared'] = df['median_income']**2
df['sqrt_population'] = np.sqrt(df['population'])
df['income_x_rooms_per_household'] = df['median_income'] * df['rooms_per_household']
df['income_x_population_density'] = df['median_income'] * df['population_density']
df['rooms_squared'] = df['total_rooms']**2
df['population_cubed'] = df['population']**(1/3)


#Aplicando el clasismo y clasidicando por ingresos
df['income_category'] = pd.cut(df['median_income'],
                               bins=[0, 2, 4, 6, 8, 15],
                               labels=[1, 2, 3, 4, 5])
df = pd.concat([df, pd.get_dummies(df['income_category'], prefix='income_cat')], axis=1)
df.drop(columns=['income_category'], inplace=True)
df['density_category'] = pd.cut(df['population_density'],
                                 bins=[0, 2, 4, 6, 8, 15],
                                 labels=[1,2,3,4,5])
df = pd.concat([df, pd.get_dummies(df['density_category'], prefix='density')], axis=1)
df.drop(columns=['density_category'], inplace=True)


df['geo_cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(df[['longitude','latitude']])
df = pd.concat([df, pd.get_dummies(df['geo_cluster'], prefix='geo')], axis=1)
df.drop(columns=['geo_cluster'], inplace=True)
central_longitude = df['longitude'].mean()
central_latitude  = df['latitude'].mean()
df['distance_to_center'] = np.sqrt((df['longitude'] - central_longitude)**2 +
                                   (df['latitude'] - central_latitude)**2)

# 5) Ahora sí
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


# más división de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)

# entrenar modelo base
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# predicciones y evaluación
predicciones = modelo.predict(X_test)

info = {
    'predicciones': predicciones,
    'y_test': y_test
}
resultados = pd.DataFrame(info)

print("R² entrenamiento:", modelo.score(X_train, y_train))
print("R² prueba:", modelo.score(X_test, y_test))

# visualizar histogramas y relaciones
sb.histplot(df['total_rooms'])
plt.show()

sb.scatterplot(x=df['median_house_value'], y=df['median_income'])
plt.show()

rmse = mean_squared_error(y_test, predicciones)
print("RMSE:", np.sqrt(rmse))

# escalado para mejorar estabilidad
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) #Ni idea pero cuando use el fit_transform supuestamente sesgo las pruebas aunque me da el mismo porcentaje por eso lo dejo


pd.DataFrame(X_test_scaled)

# modelo con datos escalados
modelo_escalado = LinearRegression()
modelo_escalado.fit(X_train_scaled, y_train)

# predicciones escaladas y evaluación
predicciones_escaled = modelo_escalado.predict(X_test_scaled)

info_escaled = {
    'predicciones': predicciones_escaled,
    'y_test': y_test
}
resultados_escaled = pd.DataFrame(info_escaled)


# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
#
# print("Intento  Random Forest:", rf.score(X_test, y_test) * 100)
print("Intento Actual:", modelo_escalado.score(X_train_scaled, y_train) * 100,"%" )
print("Intento mas alto: 77.4958485244253 %")
