import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy import integrate
import warnings
warnings.filterwarnings("ignore")

# 1. LOAD & FILTER DATA
df = pd.read_csv('dataset.csv')
pm = df[(df['pollutant_id'] == 'PM2.5') & (df['state'] == 'Maharashtra')].dropna(subset=['pollutant_avg'])
x, y, z = pm['longitude'].values, pm['latitude'].values, pm['pollutant_avg'].values
x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
print(x_min, x_max, y_min, y_max)
# 2. MATHEMATICAL FUNCTIONS & INTEGRATION
rbf = Rbf(x, y, z, function='multiquadric', smooth=2)


def P_2d(lat, lon): return max(0, rbf(lon, lat))
def P_3d(alt, lat, lon): return P_2d(lat, lon) * np.exp(-0.5 * alt) # e^(-0.5z) altitude decay

load, _ = integrate.dblquad(P_2d, x_min, x_max, lambda _: y_min, lambda _: y_max, epsabs=1, epsrel=0.1)
mass, _ = integrate.tplquad(P_3d, x_min, x_max, lambda _: y_min, lambda _: y_max, lambda _1,_2: 0, lambda _1,_2: 2, epsabs=1, epsrel=0.1)
print(f"Total Ground Load (Double): {load} | Total Airborne Mass (Triple): {mass}")

# 3. VISUALIZATION 
fig = plt.figure(figsize=(14, 6))
LON, LAT = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))
Z_surf = np.clip(rbf(LON, LAT), 0, None)

# Graph A: Double Integration (Surface Volume)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(LON, LAT, Z_surf, cmap='YlOrRd', alpha=0.9)
ax1.plot_surface(LON, LAT, np.zeros_like(Z_surf), color='gray', alpha=0.2) # Ground plane
ax1.set(title='Double Integration: Ground Volume', xlabel='Lon', ylabel='Lat', zlabel='PM2.5')

# Graph B: Triple Integration (Volumetric Cloud)
ax2 = fig.add_subplot(122, projection='3d')
L3, LA3, A3 = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40), np.linspace(0, 2, 10))
D3 = np.clip(rbf(L3, LA3), 0, None) * np.exp(-0.5 * A3)
mask = D3.flatten() > 15 # Hide empty space
ax2.scatter(L3.flatten()[mask], LA3.flatten()[mask], A3.flatten()[mask], c=D3.flatten()[mask], cmap='YlOrRd', alpha=0.3)
ax2.set(title='Triple Integration: Airborne Cloud (0-2km)', xlabel='Lon', ylabel='Lat', zlabel='Altitude (km)')

plt.tight_layout()
plt.show()
