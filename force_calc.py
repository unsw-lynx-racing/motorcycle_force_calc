"""
Created on Fri Sep  4 20:45:40 2026

@author: jaden
"""

import math

import numpy as np

NUMTESTS = 200  # number of tests to run

# Dimensions relative to rear contact patch
TOTAL_MASS = 228918.32 / 1000  # kg
GRAVITY = 9.8  # g
WHEEL_BASE = 1.3  # p
FORK_OFFSET = 0.05  # fork offset
COG = np.array([0.60293, 0, 0.65281])
FWHEELRAD = 0.578 / 2  # front wheel radius
RWHEELRAD = 0.601 / 2  # rear wheel radius
FWHEEL_THICKNESS = 0.03175  # front wheel thickness
RWHEEL_THICKNESS = 0.04445  # rear wheel thickness
FWHEEL_CENTRELINE = FWHEELRAD - FWHEEL_THICKNESS  # radius of front wheel centreline
RWHEEL_CENTRELINE = RWHEELRAD - RWHEEL_THICKNESS  # radius of rear wheel centreline
COF = 1.35  # coefficient of friction
AIR_DENS = 1.204  # sea level, 20C
CASTER_ANG = math.radians(22.5)  # caster angle  # steering angle
COLA = 0.09  # coefficient of lift*area (middle of range from cossalter)
CODA = 0.5  # coefficient of drag*area (big over estimate)
PMAX = 36 * 10**3  # max motor power
"rider inputs / variables"
STEER_ANG = math.radians(1)  # steering angle
VEL_FORWARD = np.arange(1, 51, 1)  # forward velocity
RCURVEREAR = np.linspace(
    68, 250, NUMTESTS
)  # rear wheel path curvature (just path curvature atp due to assumptions)
ROLL_ANG = np.arctan(
    VEL_FORWARD**2 / (GRAVITY * RCURVEREAR[:, None])
)  # ideal roll angle for given velocity and corner radius (rad)

CAMBER_ANG = np.arcsin(
    np.cos(STEER_ANG) * np.sin(ROLL_ANG)
    + np.cos(ROLL_ANG) * np.sin(STEER_ANG) * (np.sin(CASTER_ANG))
)  # front camber angle, negligible pitch

KINSTEER_ANG = np.arctan(np.cos(CASTER_ANG) * np.tan(STEER_ANG) / np.cos(ROLL_ANG))
# kinematic steering angle

FDRAG = 0.5 * AIR_DENS * CODA * VEL_FORWARD**2  # drag force
FLIFT = 0.5 * AIR_DENS * COLA * VEL_FORWARD**2  # aerodynamic lift force
THRUST_LEVEL_SS = FDRAG


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
    # Max load on front wheel under braking
    nmaxbrake = TOTAL_MASS * GRAVITY
    ftyremaxbrake = nmaxbrake * (WHEEL_BASE - COG[0]) / COG[2]
    rtyremaxbrake = nmaxbrake * COG[0] / COG[2]
    return (
        amax_englim,
        amax_traclim,
        amax_wheelielim,
        nmaxbrake,
        ftyremaxbrake,
        rtyremaxbrake,
    )


def ss_cornering():
    fnorm = TOTAL_MASS * GRAVITY * COG[0] / WHEEL_BASE - FDRAG * (
        COG[2] / WHEEL_BASE
    ) * np.cos(ROLL_ANG)
    rnorm = TOTAL_MASS * GRAVITY * (WHEEL_BASE - COG[0]) / WHEEL_BASE + FDRAG * (
        COG[2] / WHEEL_BASE
    ) * np.cos(ROLL_ANG)
    flateral = (
        COG[0]
        / (WHEEL_BASE * np.cos(KINSTEER_ANG))
        * VEL_FORWARD**2
        / RCURVEREAR[:, None]
        * TOTAL_MASS
    )
    rlateral = (
        (WHEEL_BASE - COG[0])
        / (WHEEL_BASE)
        * VEL_FORWARD**2
        / RCURVEREAR[:, None]
        * TOTAL_MASS
    )
    freq_cof = flateral / fnorm
    rreq_cof = rlateral / rnorm
    return fnorm, rnorm, flateral, rlateral, freq_cof, rreq_cof


lfsfnorm, lfsrnorm = level_free_stand()
ssrfnorm, ssrrnorm, ssvmax = ss_rectilinear()
(
    tra_englim,
    tra_traclim,
    tra_wheelielim,
    tra_nmaxbrake,
    tra_ftyremaxbrake,
    tra_rtyremaxbrake,
) = trans_rectilinear()
ssafnorm, ssarnorm, ssaflateral, ssarlateral, freq_cof, rreq_cof = ss_cornering()
with open("force_calc_results.txt", "w") as f:
    print(
        f"Level Free Stand:\n Front Normal Force = \n{lfsfnorm}\n Rear Normal Force = \n{lfsrnorm}",
        file=f,
    )
    print(
        f"Steady-State Rectilinear:\n Front Normal Force = \n{ssrfnorm}\n Rear Normal Force = \n{ssrrnorm}\n Maximum Velocity = \n{ssvmax}",
        file=f,
    )
    print(
        f"Transient Rectilinear:\n Maximum Engine Limited Acceleration =\n {tra_englim}\n Traction Limited Acceleration =\n {tra_traclim}\n Wheelie Limited Acceleration =\n {tra_wheelielim} \n Max Normal Load on Front and Rear Wheel Under Braking and Accel Respectively = \n {tra_nmaxbrake} \n Max Lateral Load on Front = \n {tra_ftyremaxbrake} \n Max Lateral Load on Rear = \n {tra_rtyremaxbrake}",
        file=f,
    )
    print(
        f"Steady State Cornering:\n Max Corner Frontal Load (Parallel to bike when looking from front) = \n {np.max(np.sqrt(ssafnorm**2 + ssaflateral**2)[(freq_cof < COF) & (rreq_cof < COF)])}\n Max Corner Rear Load (Parallel to bike when looking from front) = \n {np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)])}",
        file=f,
    )
    print(
        f" Max Front Normal = \n {ssafnorm[np.where(np.sqrt(ssafnorm**2 + ssaflateral**2) == np.max(np.sqrt(ssafnorm**2 + ssaflateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))]}\n Max Front Lateral = \n {ssaflateral[np.where(np.sqrt(ssafnorm**2 + ssaflateral**2) == np.max(np.sqrt(ssafnorm**2 + ssaflateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))]}\n Max Rear Normal = \n {ssarnorm[np.where(np.sqrt(ssarnorm**2 + ssarlateral**2) == np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))]}\n Max Rear Lateral = \n {ssarlateral[np.where(np.sqrt(ssarnorm**2 + ssarlateral**2) == np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))]}\n Max Rear Lateral = \n {ssarlateral[np.where(np.sqrt(ssarnorm**2 + ssarlateral**2) == np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))]}",
        file=f,
    )
    print(
        f" Max Corner Frontal Params =\n Corner Radius: \n {RCURVEREAR[np.where(np.sqrt(ssafnorm**2 + ssaflateral**2) == np.max(np.sqrt(ssafnorm**2 + ssaflateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))[0]]}\n Velocity:\n {VEL_FORWARD[np.where(np.sqrt(ssafnorm**2 + ssaflateral**2) == np.max(np.sqrt(ssafnorm**2 + ssaflateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))[1]]}\n Max Corner Rear Params =\n Corner Radius: \n {RCURVEREAR[np.where(np.sqrt(ssarnorm**2 + ssarlateral**2) == np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))[0]]}\n Velocity:\n {VEL_FORWARD[np.where(np.sqrt(ssarnorm**2 + ssarlateral**2) == np.max(np.sqrt(ssarnorm**2 + ssarlateral**2)[(freq_cof < COF) & (rreq_cof < COF)]))[1]]}",
        file=f,
    )
    print(
        f" COF:\n Front COF = \n {np.max(freq_cof[(freq_cof < COF) & (rreq_cof < COF)])}\n Rear COF = \n {np.max(rreq_cof[(freq_cof < COF) & (rreq_cof < COF)])}",
        file=f,
    )
    print(
        f"Max Roll Angle = {np.degrees(np.max(ROLL_ANG[(freq_cof < COF) & (rreq_cof < COF)]))}",
        file=f,
    )
