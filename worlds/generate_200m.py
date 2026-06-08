#!/usr/bin/env python3
import os

def generate():
    sdf = """<?xml version="1.0" ?>\n<sdf version="1.6">\n  <world name="tunnel_200">\n    <include><uri>model://ground_plane</uri></include>\n    <include><uri>model://sun</uri></include>\n"""
    
    for i in range(21):
        x = i * 10
        sdf += f"""
    <model name="segment_{i}"><static>true</static><pose>{x} 0 0 0 0 0</pose><link name="link">
        <collision name="floor"><geometry><box><size>10 6 0.2</size></box></geometry></collision>
        <visual name="floor"><pose>0 0 -0.1 0 0 0</pose><geometry><box><size>10 6 0.2</size></box></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grey</name></script></material></visual>
        <visual name="roof"><pose>0 0 3.0 0 0 0</pose><geometry><box><size>10 6 0.2</size></box></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/DarkGrey</name></script></material></visual>
        <visual name="wall_l"><pose>0 3 1.5 0 0 0</pose><geometry><box><size>10 0.2 3</size></box></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material></visual>
        <visual name="wall_r"><pose>0 -3 1.5 0 0 0</pose><geometry><box><size>10 0.2 3</size></box></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material></visual>
        <visual name="pillar_l"><pose>0 2.8 1.5 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>3</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Yellow</name></script></material></visual>
        <visual name="pillar_r"><pose>0 -2.8 1.5 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>3</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Yellow</name></script></material></visual>
    </link></model>\n"""
    
    path = os.path.expanduser("~/honeywell_ws/src/vio_pipeline/worlds/tunnel_world.world")
    with open(path, "w") as f: 
        f.write(sdf + "  </world>\n</sdf>")
    print(f"✅ World successfully generated at: {path}")

if __name__ == '__main__': 
    generate()
