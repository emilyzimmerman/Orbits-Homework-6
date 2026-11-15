
import numpy as np

# Question 3: 

# Orbital elements ---> State vector
def oe_sv(H, e, i, RA, w, theta, mu):

    # Convert angles from degrees to radians
    i = np.radians(i)
    RA = np.radians(RA)
    w = np.radians(w)
    theta = np.radians(theta)

    # Semi-latus rectum
    p = H**2 / mu

    # Position in perifocal frame
    r_pqw = np.array([
        (p * np.cos(theta)) / (1 + e * np.cos(theta)),
        (p * np.sin(theta)) / (1 + e * np.cos(theta)),
        0
    ])

    # Velocity in perifocal frame
    v_pqw = np.array([
        -(mu / H) * np.sin(theta),
        (mu / H) * (e + np.cos(theta)),
        0
    ])

    # Rotation matrix components
    cRA = np.cos(RA)
    sRA = np.sin(RA)

    cw = np.cos(w)
    sw = np.sin(w)

    ci = np.cos(i)
    si = np.sin(i)

    # Rotation matrix: Perifocal to Inertial
    Q = np.array([
        [cRA*cw - sRA*sw*ci, -cRA*sw - sRA*cw*ci, sRA*si],
        [sRA*cw + cRA*sw*ci, -sRA*sw + cRA*cw*ci, -cRA*si],
        [sw*si, cw*si, ci]
    ])

    # Rotate into inertial frame
    r = Q @ r_pqw

    # Return state vectors
    return r, mu

mu = 398600.0     
a = 7016.0         
e = 0.05          
i = 45             
RA = 0              
w = 20               
theta = 10          


#angular momentum

H = np.sqrt(mu * a * (1 - e**2))

#state vectors


r, mu = oe_sv(H, e, i, RA, w, theta, mu)


print("ORBITAL PARAMETERS")
print("-------------------------")
print("Angular Momentum:", H, "km^2/s")
print("Eccentricity:", e)
print("Inclination:", i, "degrees")
print("RAAN:", RA, "degrees")
print("Argument of Periapsis:", w, "degrees")
print("True Anomaly:", theta, "degrees")


print("\nPOSITION VECTOR")
print("-------------------------")
print("Position:", r)
print("X:", r[0], "km")
print("Y:", r[1], "km")
print("Z:", r[2], "km")
