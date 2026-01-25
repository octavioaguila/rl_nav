import time
import os
import mujoco
import mujoco.viewer
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(cwd, 'train', 'world_11_medium.xml'))
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  interval = 0
  while viewer.is_running():
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    right_wheel_ids = [m.actuator(n).id for n in ("w_rr","w_rc","w_rf")]
    left_wheel_ids = [m.actuator(n).id for n in ("w_lr","w_lc","w_lf")]

    for idx in right_wheel_ids: d.ctrl[idx] = 0.0
    for idx in left_wheel_ids: d.ctrl[idx] = 0.0

    right_angular_velocity = np.mean(d.qvel[6:9])
    left_angular_velocity = np.mean(d.qvel[9:12])

    torque_right_wheel_ids = [m.sensor(n).id for n in ("sf_rr","sf_rc","sf_rf")]
    torque_left_wheel_ids = [m.sensor(n).id for n in ("sf_lr","sf_lc","sf_lf")]

    right_torque = np.mean(d.sensordata[torque_right_wheel_ids])
    left_torque = np.mean(d.sensordata[torque_left_wheel_ids])

    # Print angular vel every 0.5 seconds
    if interval % 50 == 0:

      print(f"\nRobot Angular velocity: {d.qvel[5]}")
      print(f"Position: {d.qpos[:3]}")
      print(f"Right wheel Angular velocity: {right_angular_velocity}")
      print(f"Left wheel Angular velocity: {left_angular_velocity}")
      print(f"Right wheel Torque: {right_torque}")
      print(f"Left wheel Torque: {left_torque}")
      print(f"Torque Symmetry Error: {abs(right_torque) - abs(left_torque)}")
    interval += 1

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)