
import numpy as np
#Question 2: 

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
    r = Q @ r_pqw
    v = Q @ v_pqw

    # Return state vectors 
    return r, v, mu


# 9/29/2024 Data --- 2024 PT5
r = np.array([13.769, -6.239, 0.1697]) *1e7
v = np.array([11.629, 27.248, 0.866])
mu = 1.3271244e11 #mu of sun

H, e, i, RA, w, theta, mu = sv_oe(r, v, mu)

print("ORBITAL ELEMENTS")
print("-------------------------")
print("Angular Momentum:", H, "km^2/s")
print("Eccentricity:", e,)
print("Inclination:", i, "degrees")
print("RAAN:", RA, "degrees")
print("Argument of Periapsis:", w, "degrees")
print("True Anomaly:", theta, "degrees")
print("Gravitational Parameter (Mu):", mu, "km^3/s^2")

# Find period T of orbit: theta --> E --> M --> T
def elliptical_time(e, H, mu):

    # Periapsis
    r_p = (H**2 / mu) * (1 / (1 + e))

    # Apoapsis
    r_a = (H**2 / mu) * (1 / (1 - e))

    # Semi-major axis
    a = (r_p + r_a) / 2

    # Orbit Time 
    T = 2 * np.pi/np.sqrt(mu / a**3)
    return T

T = elliptical_time(e,H,mu)

# Convert seconds to days and years
t_days = T / (60 * 60 * 24)
T_years = T / (60 * 60 * 24 * 365.25)

print("\nTIME")
print("-------------------------")
print("Time in Days:", t_days, "days")
print("Time in Years:", T_years, "years")


## REMIND TO SELF: add units