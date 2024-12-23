"""
Using Multilayer Perceptron (MLP) for online learning to compensate
for aerodynamic disturbances and combining it with PID control for regulation.
"""

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControlLearn import DSLPIDControlLearn
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def aerodyn_pred_estimate(obs, obs_prev, thrust_prev, torques_prev,
                          ctrl, feat_full_dataset, label_full_dataset,
                          t_count, Ts, K_buffer):
    """
    Predict aerodynamic disturbances based on the quadrotor's state and control
    """

    # Feature data @ (current time): [pz, vx, vy, vz, phi, theta, psi, p, q, r](1 - by - 10)
    new_data_feat = np.hstack([obs[2], obs[10:13], obs[7:10], obs[13:16]])
    # Feature data @ (previous time): [pz, vx, vy, vz, phi, theta, psi, p, q, r](1 - by - 10)
    last_data_feat = np.hstack([obs_prev[2], obs_prev[10:13], obs_prev[7:10], obs_prev[13:16]])
    # Label data @ (previous time): [f_ax, f_ay, f_az, tau_ax, tau_ay, tau_az](6 - by - 1)
    last_data_label = ctrl.compute_label_data(obs, obs_prev, thrust_prev, torques_prev, Ts)

    # Fill the training dataset
    feat_full_dataset[t_count % K_buffer] = last_data_feat
    label_full_dataset[t_count % K_buffer] = last_data_label

    # Initialize GPR model by sklearn
    kernel = RBF(length_scale=2.0)
    gpr = GaussianProcessRegressor(kernel=kernel)
    # Train the model and make predictions
    if t_count >= K_buffer:
        # Train the model on the full dataset
        gpr.fit(feat_full_dataset, label_full_dataset)
        # Perform prediction using the model for the new data
        aerodyn_pred, sigma = gpr.predict(new_data_feat.reshape(1, -1), return_std=True)
    else:
        aerodyn_pred = np.zeros(6)

    return aerodyn_pred, last_data_label, feat_full_dataset, label_full_dataset

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                    num_drones=num_drones,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    physics=physics,
                    neighbourhood_radius=10,
                    pyb_freq=simulation_freq_hz,
                    ctrl_freq=control_freq_hz,
                    gui=gui,
                    record=record_video,
                    obstacles=obstacles,
                    user_debug_gui=user_debug_gui
                    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControlLearn(drone_model=drone) for i in range(num_drones)]

    #### Initialize the parameters ############################
    K_buffer = 20  # Dataset buffer size
    aerodyn_pred_all = []
    last_data_label_all = []
    thrust = np.zeros(num_drones)
    torques = np.zeros((num_drones, 3))
    thrust_prev = np.zeros(num_drones)
    torques_prev = np.zeros((num_drones, 3))
    action = np.zeros((num_drones, 4))
    obs_prev = np.zeros((num_drones, 20))
    aerodyn_pred = np.zeros((num_drones, 6))
    last_data_label = np.zeros((num_drones, 6))
    feat_full_dataset = np.zeros((num_drones, K_buffer, 10))
    label_full_dataset = np.zeros((num_drones, K_buffer, 6))

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        for j in range(num_drones):
            # Predict aerodynamic disturbances based on the quadrotor's state and control
            aerodyn_pred[j], last_data_label[j], feat_full_dataset[j], label_full_dataset[j] = aerodyn_pred_estimate(
                                                                obs[j], obs_prev[j], thrust_prev[j], torques_prev[j],
                                                                ctrl[j], feat_full_dataset[j], label_full_dataset[j],
                                                                i, 1./env.CTRL_FREQ, K_buffer)

            #### Compute control for the current way point #############
            action[j, :], _, _, thrust[j], torques[j, :] = ctrl[j].computeControlFromState(
                            aerodyn_pred=aerodyn_pred[j],
                            control_timestep=env.CTRL_TIMESTEP,
                            state=obs[j],
                            target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                            # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                            target_rpy=INIT_RPYS[j, :]
                            )

            #### Go to the next way point and loop #####################
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0
            
            # Store as previous frame data
            obs_prev[j] = obs[j]
            thrust_prev[j] = thrust[j]
            torques_prev[j] = torques[j]
            aerodyn_pred_all.append(aerodyn_pred)
            last_data_label_all.append(last_data_label)
            
            #### Log the simulation ####################################
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )


        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()
        logger.plot_aerodyn_pred(aerodyn_pred_all, last_data_label_all)


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
