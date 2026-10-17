#!/usr/bin/env python3
"""
Contains the main SubmarineSimulator class.
This class handles Pygame, rendering, physics, and the main game loop.
"""
import math
import random
import time
from typing import Tuple, Optional

import pygame
import numpy as np

from config import *
from world import SubmarinePhysicsState
from world import PrequalGate, PrequalMarker
# --- Import Vision ---
from data_structures import ThrusterCommands, MPU6050Readings, SensorSuite, Vision
# ---
from ai.submarine import Submarine
from collision import build_obstacles, find_collision, check_gate_crossing
from sensor_model import SensorModel
from detector_model import DetectorModel
from perception import CameraModel
from camera_geometry import segment_polygon, distance_to


class SubmarineSimulator:
    def __init__(self, submarine_ai: Submarine, width=1200, height=800, detector_mode: str = "opencv"):
        """detector_mode: 'opencv' (render the camera and run OpenCV on it),
        'perfect' (simulated detector, no noise) or 'noisy' (simulated detector with noise)."""
        assert detector_mode in ("opencv", "perfect", "noisy"), detector_mode
        self.detector_mode = detector_mode
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Submarine (AUV) Simulator")
        self.clock = pygame.time.Clock()
        self.config = SimulationConfig()
        self.prequal_config = PrequalConfig()
        self.scaleX = width * 0.7 / self.config.worldWidth
        self.scaleY = height * 0.8 / self.config.worldHeight
        self.font = pygame.font.Font(None, 36)
        self.smallFont = pygame.font.Font(None, 24)
        self.cameraSurface = pygame.Surface((self.config.cameraWidth, self.config.cameraHeight))
        self.cameraModel = CameraModel(self.config.cameraWidth, self.config.cameraHeight, self.config.cameraFov)
        self.renderCamera = True # Headless runs with a simulated detector can skip drawing the camera image
        self.overlayWidth = max(1, self.config.cameraWidth // 320) # Debug box line width, so boxes survive the downscaled view
        try:
            bg_img = pygame.image.load("BackgroundImage.jpg").convert()
            h=2*self.config.cameraHeight; w=int(bg_img.get_width()*(h/bg_img.get_height()))
            self.camera_background = pygame.transform.scale(bg_img, (w,h))
            self.camera_background_pano = pygame.Surface((w*2,h))
            self.camera_background_pano.blit(self.camera_background,(0,0)); self.camera_background_pano.blit(self.camera_background,(w,0))
        except pygame.error as e:
            print(f"Error loading background image: {e}")
            self.camera_background, self.camera_background_pano = None, None

        self.subMass = self.config.subMass
        self.subInertia_Z = self.config.subInertia_Z
        self.subInertia_Y = self.config.subInertia_Y
        self.thrusterMaxForce = self.config.thrusterMaxForce
        self.force_gravity = self.subMass * self.config.gravity
        self.force_buoyancy = self.config.subVolume * self.config.waterDensity * self.config.gravity
        self.netBuoyancyForce = self.force_buoyancy - self.force_gravity
        # Corner thrusters are angled 45 deg so their thrust lines miss the center.
        # Perpendicular distance from the center to each thrust line:
        self.yawLeverArm = 0.7071 * (self.config.submarineLength / 2 + self.config.submarineWidth / 2)
        self.poleBottomZ = self.config.worldDepth - 0.01

        self.sensorModel = SensorModel(rng=random.Random(random.random())) # Seeded from the global RNG, so runs can be replayed
        self.submarineAI = submarine_ai
        self.prequal_gate: Optional[PrequalGate] = None
        self.prequal_marker: Optional[PrequalMarker] = None
        self.resetSimulation()

    def resetSimulation(self):
        # (resetSimulation remains the same)
        self.prequal_gate = PrequalGate(
            x = self.prequal_config.GATE_X_POS, center_y = self.config.worldHeight / 2,
            z_top = self.prequal_config.GATE_DEPTH_METERS, width = self.prequal_config.GATE_WIDTH_METERS,
            height = self.prequal_config.GATE_OPENING_HEIGHT, color = self.prequal_config.GATE_COLOR
        )
        self.prequal_marker = PrequalMarker(
            x = self.prequal_config.MARKER_X_POS, y = self.config.worldHeight / 2,
            z_top = -self.prequal_config.POLE_ABOVE_SURFACE_METERS,
            z_bottom = self.config.worldDepth - 0.01,
            radius = self.prequal_config.MARKER_DIAMETER_METERS / 2, color = self.prequal_config.MARKER_COLOR
        )
        self.obstacles = build_obstacles(self.prequal_gate, self.prequal_marker,
                                         self.config.gatePoleRadius, self.poleBottomZ)
        self.detector = None
        if self.detector_mode != "opencv":
            self.detector = DetectorModel(self.cameraModel, self.prequal_gate, self.prequal_marker,
                                          self.poleBottomZ, self.config.gatePoleRadius,
                                          noisy=self.detector_mode == "noisy",
                                          rng=random.Random(random.random()))
        self.lastCollision = None
        self.timeSinceContact = float('inf')
        self.collisionCount = 0
        self.gatePasses = 0
        start_heading = 0
        start_depth = self.prequal_config.START_Z_POS

        self.subPhysics = SubmarinePhysicsState(
            x = self.prequal_config.START_X_POS, y = self.config.worldHeight / 2 + random.uniform(-0.5, 0.5),
            z = start_depth,
            heading = start_heading, pitch = 0.0
        )
        self.sensorModel.reset()
        self.submarineAI.reset()
        self.startTime = time.time()
        self.running, self.paused = True, False
        self.lastThrusterCommands, self.last_imu_readings = ThrusterCommands(), MPU6050Readings()

    def worldToScreen(self, x, y):
        # (Remains the same)
        return int(x*self.scaleX+50), int((self.config.worldHeight-y)*self.scaleY+50)

    def handleInput(self):
        # (Remains the same)
        for event in pygame.event.get():
             if event.type == pygame.QUIT: self.running = False
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_SPACE: self.paused = not self.paused
                 elif event.key == pygame.K_r: self.resetSimulation()

    def applyPhysics(self, dt, commands: ThrusterCommands):
        # (Remains the same - 6-DoF Physics)
        f_hfl = commands.hfl * self.thrusterMaxForce
        f_hfr = commands.hfr * self.thrusterMaxForce
        f_hal = commands.hal * self.thrusterMaxForce
        f_har = commands.har * self.thrusterMaxForce
        f_vf = commands.vf * self.thrusterMaxForce
        f_vr = commands.vr * self.thrusterMaxForce
        cos_45 = 0.7071
        thrust_surge = (f_hfl + f_hfr + f_hal + f_har) * cos_45
        thrust_sway  = (f_hfl - f_hfr - f_hal + f_har) * cos_45
        thrust_heave = f_vf + f_vr
        sub_l_half = self.config.submarineLength / 2
        thrust_yaw = (f_hfl - f_hfr + f_hal - f_har) * self.yawLeverArm
        thrust_pitch = (f_vf - f_vr) * sub_l_half
        h_rad = math.radians(self.subPhysics.heading)
        p_rad = math.radians(self.subPhysics.pitch)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        cos_p, sin_p = math.cos(p_rad), math.sin(p_rad)
        vx_w, vy_w, vz_w = self.subPhysics.velocity_x, self.subPhysics.velocity_y, self.subPhysics.velocity_z
        vel_surge = vx_w * cos_h + vy_w * sin_h
        vel_sway  = -vx_w * sin_h + vy_w * cos_h
        vel_heave = vz_w
        drag_surge = -self.config.surgeDragCoeff * vel_surge * abs(vel_surge)
        drag_sway  = -self.config.swayDragCoeff * vel_sway * abs(vel_sway)
        drag_heave = -self.config.heaveDragCoeff * vel_heave * abs(vel_heave)
        drag_yaw   = -self.config.angularDragCoeff_Z * self.subPhysics.angular_velocity_z**2 * np.sign(self.subPhysics.angular_velocity_z)
        drag_pitch = -self.config.angularDragCoeff_Y * self.subPhysics.angular_velocity_y**2 * np.sign(self.subPhysics.angular_velocity_y)
        total_force_surge = thrust_surge + drag_surge
        total_force_sway  = thrust_sway + drag_sway
        total_force_heave = thrust_heave + drag_heave
        total_torque_yaw   = thrust_yaw + drag_yaw
        total_torque_pitch = thrust_pitch + drag_pitch
        f_x_pitched = total_force_surge * cos_p - total_force_heave * sin_p
        f_z_pitched = total_force_surge * sin_p + total_force_heave * cos_p
        fx = f_x_pitched * cos_h - total_force_sway * sin_h
        fy = f_x_pitched * sin_h + total_force_sway * cos_h
        fz = f_z_pitched - self.netBuoyancyForce
        ax = fx / self.subMass
        ay = fy / self.subMass
        az = fz / self.subMass
        angular_accel_z = total_torque_yaw / self.subInertia_Z
        angular_accel_y = total_torque_pitch / self.subInertia_Y
        self.subPhysics.velocity_x += ax * dt
        self.subPhysics.velocity_y += ay * dt
        self.subPhysics.velocity_z += az * dt
        self.subPhysics.angular_velocity_z += angular_accel_z * dt
        self.subPhysics.angular_velocity_y += angular_accel_y * dt
        imu_accel_surge = ax * cos_h + ay * sin_h
        imu_accel_sway = -ax * sin_h + ay * cos_h
        imu_accel_heave = az
        self.last_imu_readings = MPU6050Readings(
            accel_x=imu_accel_sway, accel_y=imu_accel_surge, accel_z=imu_accel_heave,
            gyro_z=self.subPhysics.angular_velocity_z,
            gyro_y=self.subPhysics.angular_velocity_y
        )
        prev = (self.subPhysics.x, self.subPhysics.y, self.subPhysics.z,
                self.subPhysics.heading, self.subPhysics.pitch)
        self.subPhysics.x += self.subPhysics.velocity_x * dt
        self.subPhysics.y += self.subPhysics.velocity_y * dt
        self.subPhysics.z += self.subPhysics.velocity_z * dt
        self.subPhysics.heading = (self.subPhysics.heading + math.degrees(self.subPhysics.angular_velocity_z * dt)) % 360
        self.subPhysics.pitch = (self.subPhysics.pitch + math.degrees(self.subPhysics.angular_velocity_y * dt))
        self.subPhysics.pitch = np.clip(self.subPhysics.pitch, -90, 90)
        margin = 0.5
        self.subPhysics.x = np.clip(self.subPhysics.x, margin, self.config.worldWidth - margin)
        self.subPhysics.y = np.clip(self.subPhysics.y, margin, self.config.worldHeight - margin)
        self.subPhysics.z = np.clip(self.subPhysics.z, 0.0, self.config.worldDepth - 0.2)
        self._resolveCollisions(prev, dt)
        if self.prequal_gate:
            crossing = check_gate_crossing(self.prequal_gate, prev[0], prev[1], prev[2],
                                           self.subPhysics.x, self.subPhysics.y, self.subPhysics.z,
                                           self.poleBottomZ)
            if crossing:
                if crossing.inside_opening: self.gatePasses += 1
                where = "through the gate" if crossing.inside_opening else "outside the gate opening"
                print(f"Gate plane crossed {'forward' if crossing.direction > 0 else 'backward'} {where} "
                      f"(y={crossing.y:.2f}, z={crossing.z:.2f})")

    def _resolveCollisions(self, prev, dt):
        """ On contact, return the sub to its last clear pose and stop it. """
        s = self.subPhysics
        hit = find_collision(s.x, s.y, s.z, s.heading, self.config.submarineLength,
                             self.config.submarineWidth, self.config.submarineHeight, self.obstacles)
        clear_for = self.timeSinceContact
        self.timeSinceContact = 0.0 if hit else self.timeSinceContact + dt
        if hit:
            # A new collision, unless it's the same contact bumping again within half a second
            if clear_for > 0.5:
                self.collisionCount += 1
                print(f"Collision with {hit.name} at x={s.x:.2f}, y={s.y:.2f}, z={s.z:.2f}")
            s.x, s.y, s.z, s.heading, s.pitch = prev
            s.velocity_x = s.velocity_y = s.velocity_z = 0.0
            s.angular_velocity_z = s.angular_velocity_y = 0.0
        self.lastCollision = hit

    def project3D(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        # (Remains the same)
        dx,dy,dz = world_pos[0]-self.subPhysics.x, world_pos[1]-self.subPhysics.y, world_pos[2]-self.subPhysics.z
        h,p = math.radians(-self.subPhysics.heading), math.radians(-self.subPhysics.pitch)
        ch,sh,cp,sp = math.cos(h),math.sin(h),math.cos(p),math.sin(p)
        x_yaw, y_yaw = dx*ch-dy*sh, dx*sh+dy*ch
        cz,cy,cx = x_yaw*cp+dz*sp, x_yaw*sp-dz*cp, y_yaw
        if cz < 0.2: return None
        w,h = self.cameraSurface.get_size()
        f = w/(2*math.tan(math.radians(self.config.cameraFov/2)))
        return int(w/2-f*(cx/cz)), int(h/2-f*(cy/cz)), math.hypot(dx,dy,dz)


    def generateCameraView(self):
        """ Draws the camera image: background, then every object at its real size, far to near. """
        w,h = self.cameraSurface.get_size()
        if self.camera_background_pano:
             bg_w,bg_h = self.camera_background.get_size()
             x_off = (self.subPhysics.heading/360)*bg_w
             px_per_deg = h / 120.0 # Tuned as 2 px/deg at 240 tall
             y_off = np.clip(((bg_h-h)/2)-(self.subPhysics.pitch*px_per_deg), 0, bg_h-h)
             self.cameraSurface.blit(self.camera_background_pano, (-x_off, -y_off))
        else:
             self.cameraSurface.fill(WATER_COLOR)
             hp = self.project3D((self.subPhysics.x+20, self.subPhysics.y, self.config.worldDepth))
             if hp: pygame.draw.rect(self.cameraSurface, POOL_FLOOR_COLOR, (0,hp[1],w,h))

        # (p1, p2, radius, color) for each object: the same geometry the detector uses.
        segments = []
        if self.prequal_gate:
             g = self.prequal_gate
             r = self.config.gatePoleRadius
             left = (g.x, g.center_y - g.width / 2)
             right = (g.x, g.center_y + g.width / 2)
             segments.append(((*left, g.z_top), (*right, g.z_top), r, GRAY))        # Top bar
             segments.append(((*left, g.z_top), (*left, self.poleBottomZ), r, g.color))
             segments.append(((*right, g.z_top), (*right, self.poleBottomZ), r, g.color))
        if self.prequal_marker:
             m = self.prequal_marker
             segments.append(((m.x, m.y, m.z_top), (m.x, m.y, m.z_bottom), m.radius, m.color))

        drawable = []
        for p1, p2, radius, color in segments:
             poly = segment_polygon(self.subPhysics, self.cameraModel, p1, p2, radius)
             if poly:
                 drawable.append((distance_to(self.subPhysics, p1, p2), color, poly))
        drawable.sort(key=lambda d: d[0], reverse=True)
        limit = 4 * max(w, h) # Keep huge coordinates of very close objects in a range pygame handles
        for _, color, poly in drawable:
             pts = [(min(max(x, -limit), limit), min(max(y, -limit), limit)) for x, y in poly]
             pygame.draw.polygon(self.cameraSurface, color, pts)

    def render(self):
        # (Remains the same)
        self.screen.fill(LIGHT_BLUE)
        pygame.draw.rect(self.screen, BLACK, (40,40,int(self.config.worldWidth*self.scaleX+20),int(self.config.worldHeight*self.scaleY+20)), 2)
        if self.prequal_gate: g = self.prequal_gate; p1 = self.worldToScreen(g.x, g.center_y - g.width / 2); p2 = self.worldToScreen(g.x, g.center_y + g.width / 2); pygame.draw.line(self.screen, g.color, p1, p2, 4)
        if self.prequal_marker: m = self.prequal_marker; pos_2d = self.worldToScreen(m.x, m.y); radius_scaled = max(2, int(m.radius * self.scaleX)); pygame.draw.circle(self.screen, m.color, pos_2d, radius_scaled); pygame.draw.circle(self.screen, BLACK, pos_2d, radius_scaled, 1)
        subPos = self.worldToScreen(self.subPhysics.x, self.subPhysics.y); hRad, cos_h, sin_h = math.radians(self.subPhysics.heading), math.cos(math.radians(self.subPhysics.heading)), math.sin(math.radians(self.subPhysics.heading)); pvc_s = self.config.submarineWidth*self.scaleX/2
        corners = [(-pvc_s,-pvc_s), (pvc_s,-pvc_s), (pvc_s,pvc_s), (-pvc_s,pvc_s)]; rotated = [(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in corners]; pygame.draw.polygon(self.screen,YELLOW,rotated,4)
        box_w,box_l=0.127*self.scaleY/2,self.config.submarineLength*self.scaleX/2; box_corners=[(-box_l,-box_w),(box_l,-box_w),(box_l,box_w),(-box_l,box_w)]; rotated_box=[(subPos[0]+dx*cos_h-dy*sin_h,subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in box_corners]; pygame.draw.polygon(self.screen,CONTROL_BOX_GRAY,rotated_box)
        arrow_pts = [(box_l,-box_w),(box_l,box_w),(box_l+0.2*self.scaleX,0)]; rotated_arrow=[(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in arrow_pts]; pygame.draw.polygon(self.screen, YELLOW, rotated_arrow)
        self._renderUi(); scaled_camera = pygame.transform.scale(self.cameraSurface, (400, 300)); self.screen.blit(scaled_camera, (self.width-420, 20)); pygame.draw.rect(self.screen, BLACK, (self.width-420, 20, 400, 300), 2); pygame.display.flip()

    def _drawThrusterBar(self, x, y, label, value):
        # (Remains the same)
        bar_w, bar_h = 25, 80; center_y = y + bar_h / 2; pygame.draw.rect(self.screen, GRAY, (x, y, bar_w, bar_h), 2)
        if value > 0: pygame.draw.rect(self.screen, GREEN, (x + 1, center_y - value * bar_h / 2, bar_w - 2, value * bar_h / 2))
        elif value < 0: pygame.draw.rect(self.screen, RED, (x + 1, center_y, bar_w - 2, abs(value) * bar_h / 2))
        pygame.draw.line(self.screen, BLACK, (x, center_y), (x + bar_w, center_y), 1); label_surf = self.smallFont.render(label, True, BLACK); self.screen.blit(label_surf, (x + bar_w / 2 - label_surf.get_width() / 2, y + bar_h + 5))

    def _renderUi(self):
        # (Remains the same)
        y = 20; title = self.font.render("Autonomous Submarine Simulator (AUV)", True, BLACK); self.screen.blit(title, (20,y)); y+=40
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y)
        task,state = self.submarineAI.get_current_task_name(), self.submarineAI.get_current_state_name()
        stats=[f"Time: {time.time()-self.startTime:.1f}s", f"Task: {task}", f"State: {state}",
               f"Speed (XY): {speed:.2f} m/s", f"Vel Z: {self.subPhysics.velocity_z:.2f} m/s",
               f"Heading: {self.subPhysics.heading:.1f}°", f"Pitch: {self.subPhysics.pitch:.1f}°",
               f"Depth: {self.subPhysics.z:.2f} m",
               f"Gate passes: {self.gatePasses}  Collisions: {self.collisionCount}"]
        for s in stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=20
        y+=10; imu = self.last_imu_readings
        imu_stats=["IMU:",
                   f" Accel Y(surge): {imu.accel_y: .2f} m/s²",
                   f" Accel X(sway): {imu.accel_x: .2f} m/s²",
                   f" Accel Z(heave): {imu.accel_z: .2f} m/s²",
                   f" Gyro Z(yaw): {math.degrees(imu.gyro_z): .1f}°/s",
                   f" Gyro Y(pitch): {math.degrees(imu.gyro_y): .1f}°/s"]
        for s in imu_stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=18
        y = self.height - 80; controls=["Controls:", "R - Reset", "SPACE - Pause"]
        for c in controls: self.screen.blit(self.smallFont.render(c,True,BLACK),(20,y)); y+=18
        tx,ty = self.width-420,350; self.screen.blit(self.smallFont.render("Thruster Output:",True,BLACK),(tx,ty)); ty+=25
        tc=self.lastThrusterCommands
        h_labels=[("HFL",tc.hfl),("HFR",tc.hfr),("HAL",tc.hal),("HAR",tc.har)]
        v_labels=[("VF",tc.vf),("VR",tc.vr)]
        for i,(l,v) in enumerate(h_labels): self._drawThrusterBar(tx+i*40,ty,l,v)
        for i,(l,v) in enumerate(v_labels): self._drawThrusterBar(tx+180+i*40,ty,l,v)

    def run(self):
        while self.running:
             dt = self.clock.tick(60) / 1000.0;
             if dt > 0.1: dt = 0.1
             self.handleInput()
             if not self.paused:
                 self.step(dt)
             self.render()
        pygame.quit()

    def step(self, dt: float):
        """ One sense / think / move cycle, without drawing the main window. """
        if self.renderCamera: self.generateCameraView()
        detections = self.detector.detect(self.subPhysics, dt) if self.detector else None
        measured = self.sensorModel.read(self.subPhysics, self.last_imu_readings, dt)
        sensors = SensorSuite(camera_image=self.cameraSurface, depth=measured.depth,
                              heading=measured.heading, pitch=measured.pitch,
                              imu=measured.imu, depth_rate=measured.depth_rate,
                              x=self.subPhysics.x, y=self.subPhysics.y,
                              velocity_x=self.subPhysics.velocity_x, velocity_y=self.subPhysics.velocity_y,
                              angular_velocity_y=self.subPhysics.angular_velocity_y,
                              velocity_z=self.subPhysics.velocity_z,
                              detections=detections)
        thrusterCommands, vision_results = self.submarineAI.update(dt, sensors)
        if thrusterCommands.pause_simulation: self.paused = True
        self.lastThrusterCommands = thrusterCommands
        # --- Vision Debug Drawing (Remains the same) ---
        for pole in vision_results.red_blobs:
             pygame.draw.rect(self.cameraSurface, ORANGE,
                              (pole['min_x'], pole['min_y'], pole['width'], pole['height']), self.overlayWidth)
        gate_pair = vision_results.get_gate_pair()
        if gate_pair:
            left_pole, right_pole = gate_pair
            min_x = left_pole['min_x']
            max_x = right_pole['max_x']
            min_y = min(left_pole['min_y'], right_pole['min_y'])
            max_y = max(left_pole['max_y'], right_pole['max_y'])
            pygame.draw.rect(self.cameraSurface, YELLOW, (min_x, min_y, max_x - min_x, max_y - min_y), self.overlayWidth)
        best_pole = vision_results.get_best_pole()
        if best_pole:
             pygame.draw.rect(self.cameraSurface, GREEN,
                              (best_pole['min_x'], best_pole['min_y'], best_pole['width'], best_pole['height']), 2 * self.overlayWidth)
        self.applyPhysics(dt, thrusterCommands)