
import numpy as np
#Question 1: 

# How small my numbers can be
tolerance = 1e-10

# Helper function to keep numbers in defined cos
def acos_check(x):
    return np.arccos(np.clip(x, -1, 1))


# State vector ---> Orbital elements
def sv_oe(r, v, mu):

    # State vector info
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    # Magnitudes
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)

    # Angular momentum
    h = np.cross(r, v)
    H = np.linalg.norm(h)

    # Eccentricity
    e = (np.cross(v, h)/mu) - r/R
    e_mag = np.linalg.norm(e)

    # Inclination
    i = acos_check(h[2]/H)

    # Node for right ascension
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    N = np.linalg.norm(n)

    # Right Ascension (RA)
    if N > tolerance:
        RA = acos_check(n[0]/N)
        if n[1] < 0:
            RA = 2 * np.pi - RA
    else:
        # Equatorial orbit: RA is undefined
        RA = 0

    RA_degree = np.degrees(RA)

    # Periapsis
    if e_mag > tolerance:
        if N > tolerance:
            w = acos_check(np.dot(n, e)/(N*e_mag))
            if e[2] < 0:
                w = 2 * np.pi - w
        else:
            # Equatorial orbit
            w = np.arctan2(e[1], e[0]) % (2*np.pi)

        w_degree = np.degrees(w)
    else:
        raise ValueError("Periapsis is undefined for a circular orbit.")

    # True anomaly
    theta = acos_check(np.dot(e, r)/(e_mag*R))
    if np.dot(r, v) < 0:
        theta = 2 * np.pi - theta

    theta_degree = np.degrees(theta)

    return H, e_mag, np.degrees(i), RA_degree, w_degree, theta_degree, mu


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
    #REMINDER: Python (numpy): matrix multiplication uses @. If you use *, you are doing element-wise multiplication.
    r = Q @ r_pqw
    v = Q @ v_pqw

    # Return state vectors 
    return r, v, mu


# MARS TEST
r = np.array([4000, 2000, 1000])
v = np.array([-1.5, 2.0, 0.5])   
mu = 42828


H, e, i, RA, w, theta, mu = sv_oe(r, v, mu)

print("ORBITAL ELEMENTS")
print("-------------------------")
print("Angular Momentum:", H, "km^2/s")
print("Eccentricity:", e, "(dimensionless)")
print("Inclination:", i, "degrees")
print("RAAN:", RA, "degrees")
print("Argument of Periapsis:", w, "degrees")
print("True Anomaly:", theta, "degrees")
print("Gravitational Parameter (Mu):", mu, "km^3/s^2")

# Test other function
r_new, v_new, mu = oe_sv(H, e, i, RA, w, theta, mu)

print("\nNEW STATE VECTORS")
print("-------------------------")
print("Original Position:", r, "km")
print("New Position:", r_new, "km")

print("Original Velocity:", v, "km/s")
print("New Velocity:", v_new, "km/s")

# Compare original and new vectors
print("\nERROR CHECK")
print("-------------------------")

print("Position matches:",
      np.allclose(r, r_new, atol=1e-5))

print("Velocity matches:",
      np.allclose(v, v_new, atol=1e-5))