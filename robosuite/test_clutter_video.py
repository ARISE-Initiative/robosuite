import robosuite as suite
import numpy as np
import cv2
import os

# We need this import to make sure the env is registered
import robosuite.environments.manipulation.pick_place_clutter

class ClutterTest:
    def __init__(self):
        print("Creating environment...")
        self.env = suite.make(
            env_name="PickPlaceClutter", # <-- Your new env
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True, # <-- Must be True
            use_camera_obs=True,         # <-- Must be True
            camera_names="agentview",
            camera_heights=512,
            camera_widths=512,
            control_freq=20,
            horizon=200,
        )
        self.obs = self.env.reset()
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self.obs["robot0_eef_quat"].copy()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self.trial_info_text = "" # Added this attribute

    #=====================================================================
    # Video Saving Helpers (Copied from your run.py)
    #=====================================================================
    def _video_frame(self, text=None):
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return

        H, W = self._rec["H"], self._rec["W"]
        cam = self._rec["camera"]

        # Render
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)

        # Ensure dtype + contiguity for VideoWriter
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        # Optional overlays
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        if self.trial_info_text:
            y = H - 10
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)

        # Track frames-written to catch “empty” files
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="clutter_test.mp4", fps=30, H=512, W=512, camera_name="agentview"):
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                break
        if not self._rec["on"]:
            raise RuntimeError(f"Failed to open VideoWriter for {path}. Install H.264 support or try .avi with XVID.")

        # Warm up the renderer once after previous resets
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)

        # Seed file with a few frames so it’s never near-empty
        for _ in range(3):
            self._video_frame("start")

    def _video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False
            
    def run_test(self):
        print("Starting video test...")
        self._video_start() # Start recording
        
        print("Rendering 200 steps (10 seconds) of video...")
        for i in range(200):
            if i % 20 == 0:
                print(f"Step {i}...")
                
            self.trial_info_text = f"Step: {i} / 200"
            
            # Take a "do nothing" action
            action = [0] * self.env.action_dim
            self.obs, reward, done, info = self.env.step(action)
            
            # Save the frame
            self._video_frame("Testing Clutter Env")
            
            if done:
                break
                
        # Clean up
        self._video_stop()
        self.env.close()
        print(f"\nTest complete. Video saved to clutter_test.mp4")
        print("Please check the video file to confirm your objects are in the bin.")


if __name__ == "__main__":
    # Need to import this for R.from_quat
    from scipy.spatial.transform import Rotation as R
    
    print("Initializing test...")
    test = ClutterTest()
    test.run_test()

