import numpy as np

nx, ny = 360, 181

# Wind U/V (arbitrary values, e.g., sine/cosine pattern or random)
wind_u = np.random.uniform(-10, 10, size=(ny, nx)).astype(np.float32)
wind_v = np.random.uniform(-10, 10, size=(ny, nx)).astype(np.float32)
np.save("/data/wind_u_dummy.npy", wind_u)
np.save("/data/wind_v_dummy.npy", wind_v)

# Temperature (e.g., values between -30 and 50 Celsius)
temperature = np.random.uniform(-30, 50, size=(ny, nx)).astype(np.float32)
np.save("/data/temperature_dummy.npy", temperature)

# Rain (e.g., values between 0 and 100 mm)
rain = np.random.uniform(0, 100, size=(ny, nx)).astype(np.float32)
np.save("/data/rain_dummy.npy", rain)
