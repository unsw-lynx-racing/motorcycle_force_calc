"""
Created on Fri Sep  4 20:45:40 2026

@author: jaden
"""

import math

import numpy as np

NUMTESTS = 30  # number of tests to run

# Dimensions relative to rear contact patch

BIKE_MASS = 120  # m
RIDER_MASS = 100
TOTAL_MASS = BIKE_MASS + RIDER_MASS
GRAVITY = 9.8  # g
WHEEL_BASE = 1.3  # p
FORK_OFFSET = 0.05  # fork offset
# Bike COG assuming centred
BIKECOG = np.array([0.65, 0, 0.895 / 2])  # x, y, z
# Rider COG is just placed randomly, should be changed based on erg study
RIDERCOG = np.array([0.50, 0, 1.0])  # x, y, z
COG = (BIKE_MASS * BIKECOG + RIDER_MASS * RIDERCOG) / (TOTAL_MASS)
FWHEELRAD = 0.578 / 2  # front wheel radius
RWHEELRAD = 0.601 / 2  # rear wheel radius
FWHEEL_THICKNESS = 0.03175  # front wheel thickness
RWHEEL_THICKNESS = 0.04445  # rear wheel thickness
FWHEEL_CENTRELINE = FWHEELRAD - FWHEEL_THICKNESS  # radius of front wheel centreline
RWHEEL_CENTRELINE = RWHEELRAD - RWHEEL_THICKNESS  # radius of rear wheel centreline
COF = 1.5  # coefficient of friction
AIR_DENS = 1.204  # sea level, 20C
CASTER_ANG = math.radians(22.5)  # caster angle  # steering angle
COLA = 0.09  # coefficient of lift*area (middle of range from cossalter)
CODA = 0.5  # coefficient of drag*area (big over estimate)
PMAX = 36 * 10**3  # max motor power
"rider inputs / variables"
ROLL_ANG = np.linspace(0, math.pi / 3, NUMTESTS)  # roll angle (rad)
STEER_ANG = math.radians(1)  # steering angle
VEL_FORWARD = 30  # forward velocity
beta_dash = CASTER_ANG + np.arctan(
    (np.sin(STEER_ANG) * np.tan(ROLL_ANG) - math.sin(CASTER_ANG) * np.cos(STEER_ANG))
    / math.cos(CASTER_ANG)
)
CAMBER_ANG = np.arcsin(
    np.cos(STEER_ANG) * np.sin(ROLL_ANG)
    + np.cos(ROLL_ANG) * np.sin(STEER_ANG) * (np.sin(CASTER_ANG))
)  # front camber angle, negligible pitch
c1 = (
    FORK_OFFSET * math.sin(CASTER_ANG) * (1 - np.cos(STEER_ANG))
    + RWHEEL_THICKNESS
    - FWHEEL_THICKNESS
)
c2 = FWHEEL_CENTRELINE * (
    math.cos(CASTER_ANG) * np.cos(beta_dash - CASTER_ANG)
    - np.cos(STEER_ANG) * math.sin(CASTER_ANG) * np.sin(CAMBER_ANG - CASTER_ANG)
    - 1
)
c3 = FORK_OFFSET * np.sin(STEER_ANG) + FWHEEL_CENTRELINE * np.sin(STEER_ANG) * np.sin(
    beta_dash - CASTER_ANG
)
c4 = WHEEL_BASE - FORK_OFFSET * math.cos(CASTER_ANG) * (1 - np.cos(STEER_ANG))
c5 = FWHEEL_CENTRELINE * (
    math.sin(CASTER_ANG) * np.cos(beta_dash - CASTER_ANG)
    + np.cos(STEER_ANG) * math.cos(CASTER_ANG) * np.sin(beta_dash - CASTER_ANG)
)
mu = (
    (c1 + c2) * np.cos(ROLL_ANG)
    + c3 * np.sin(ROLL_ANG)
    + FWHEEL_THICKNESS
    - RWHEEL_THICKNESS
) / ((c4 + c5) * np.cos(ROLL_ANG))  # driving traction coefficient
KINSTEER_ANG = np.arctan(
    (np.sin(STEER_ANG) * np.cos(CASTER_ANG + mu))
    / (
        np.cos(ROLL_ANG) * (np.cos(STEER_ANG))
        - np.sin(ROLL_ANG) * np.sin(STEER_ANG) * np.sin(CASTER_ANG + mu)
    )
)  # kinematic steering angle
x_Pf = (c1 + c2) * np.sin(mu) + (c4 + c5) * np.cos(mu)
y_Pf = (
    (-(c1 + c2) * np.cos(mu) + (c4 + c5) * np.sin(mu)) * np.sin(ROLL_ANG)
    + c3 * np.cos(ROLL_ANG)
    - (FWHEEL_THICKNESS - RWHEEL_THICKNESS)
)
C = np.tan(STEER_ANG) / (x_Pf + y_Pf * np.tan(STEER_ANG))  # path curvature

FDRAG = 0.5 * AIR_DENS * CODA * VEL_FORWARD**2  # drag force
FAERO = 0.5 * AIR_DENS * COLA * VEL_FORWARD**2  # aerodynamic force
THRUST_LEVEL_SS = FDRAG
RCURVEREAR = WHEEL_BASE / np.tan(KINSTEER_ANG)


def level_free_stand():
    fnorm = TOTAL_MASS * GRAVITY * COG[0] / WHEEL_BASE
    rnorm = TOTAL_MASS * GRAVITY * (WHEEL_BASE - COG[0]) / WHEEL_BASE

    return fnorm, rnorm


def ss_rectilinear():
    fnorm = TOTAL_MASS * GRAVITY * COG[0] / WHEEL_BASE - THRUST_LEVEL_SS * (
        COG[2] / WHEEL_BASE
    )

    rnorm = TOTAL_MASS * GRAVITY * (
        WHEEL_BASE - COG[0]
    ) / WHEEL_BASE + THRUST_LEVEL_SS * (COG[2] / WHEEL_BASE)
    vmax = np.sqrt(
        (TOTAL_MASS * GRAVITY)
        / (
            0.5 * AIR_DENS * CODA * (COG[2] / WHEEL_BASE)
            + 0.5 * AIR_DENS * COLA * (COG[0] / WHEEL_BASE)
        )
        * (COG[0] / WHEEL_BASE)
    )
    return fnorm, rnorm, vmax


def trans_rectilinear():
    amax_englim = (PMAX / VEL_FORWARD - FDRAG) / TOTAL_MASS
    amax_traclim = (COF * GRAVITY * (WHEEL_BASE - COG[0]) / WHEEL_BASE) / (
        1 - COF * COG[2] / WHEEL_BASE
    ) - FDRAG / TOTAL_MASS
    amax_wheelielim = GRAVITY * (COG[0] / COG[2]) - FDRAG / TOTAL_MASS
    return amax_englim, amax_traclim, amax_wheelielim
    """trans_rectilinear_amax = min(amax_englim, amax_traclim, amax_wheelielim)
    if trans_rectilinear_amax == amax_englim:
        return amax_englim, "Engine limited"
    elif trans_rectilinear_amax == amax_traclim:
        return amax_traclim, "Traction limited"
    else:
        return amax_wheelielim, "Wheelie limited"""


def ss_cornering():
    fnorm = TOTAL_MASS * GRAVITY * COG[0] / WHEEL_BASE - FAERO * (
        COG[2] / WHEEL_BASE
    ) * np.cos(ROLL_ANG)
    rnorm = TOTAL_MASS * GRAVITY * (WHEEL_BASE - COG[0]) / WHEEL_BASE + FAERO * (
        COG[2] / WHEEL_BASE
    ) * np.cos(ROLL_ANG)
    flateral = fnorm / (GRAVITY * np.cos(KINSTEER_ANG)) * (VEL_FORWARD**2 / RCURVEREAR)
    rlateral = rnorm / GRAVITY * (VEL_FORWARD**2 / RCURVEREAR)
    freq_cof = flateral / fnorm
    rreq_cof = rlateral / rnorm
    return fnorm, rnorm, flateral, rlateral, freq_cof, rreq_cof


lfsfnorm, lfsrnorm = level_free_stand()
ssrfnorm, ssrrnorm, ssvmax = ss_rectilinear()
tra_englim, tra_traclim, tra_wheelielim = trans_rectilinear()
ssafnorm, ssarnorm, ssaflateral, ssarlateral, freq_cof, rreq_cof = ss_cornering()
with open("force_calc_results.txt", "w") as f:
    print(KINSTEER_ANG, file=f)
    print(mu, file=f)
    print(
        f"Level Free Stand:\n Front Normal Force = \n{lfsfnorm}\n Rear Normal Force = \n{lfsrnorm}",
        file=f,
    )
    print(
        f"Steady-State Rectilinear:\n Front Normal Force = \n{ssrfnorm}\n Rear Normal Force = \n{ssrrnorm}\n Maximum Velocity = \n{ssvmax}",
        file=f,
    )
    print(
        f"Transient Rectilinear:\n Maximum Engine Limited Acceleration =\n {tra_englim}\n Traction Limited Acceleration =\n {tra_traclim}\n Wheelie Limited Acceleration =\n {tra_wheelielim}",
        file=f,
    )
    print(
        f"Steady-State Cornering:\n Front Normal Force =\n {ssafnorm}\n Rear Normal Force =\n {ssarnorm}\n Front Lateral Force =\n {ssaflateral}\n Rear Lateral Force =\n {ssarlateral}\n Front Coefficient of Friction =\n {freq_cof}\n Rear Coefficient of Friction =\n {rreq_cof}",
        file=f,
    )
