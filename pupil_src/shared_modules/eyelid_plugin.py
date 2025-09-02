import os
import cv2
import torch
import numpy as np
from pyglui import ui, font
from pyglui.cygl.utils import Named_Texture
from plugin import Plugin
from gl_utils import draw_gl_points_norm
from .eyelid_detector import (
    AdaptiveEyeNet6Layers,
    calculate_ear,
    apply_camera_rotation,
)


class Eyelid_Plugin(Plugin):
    """
    A plugin to detect eyelid landmarks, apply camera rotation, and calculate EAR.
    """

    def __init__(
        self,
        g_pool,
        menu_name="Eyelid Detector",
        model_path_below="/Users/witlab/Documents/2025-projects/EyeWithShut-Study4/after_finetuning_belowcameraview_best.pth",
        model_path_side="/Users/witlab/Documents/2025-projects/EyeWithShut-Study4/after_finetuning_sidecameraview_best.pth",
    ):
        super().__init__(g_pool)

        self.menu_name = menu_name
        self.model_path_below = model_path_below
        self.model_path_side = model_path_side

        # UI elements
        self.menu = None
        self.icon = None
        self.camera_position = "left_below"
        self.model_enabled = False
        self.ear_enabled = False
        self.ear_value = 0.0

        # Model
        self.device = torch.device("cpu")
        self.model = None

        # Frame and landmark data
        self.rotated_frame = None
        self.landmarks = None
        self.frame_texture = None

        # Text rendering
        self.font = None
        self.text_texture = None

    def init_ui(self):
        self.add_menu()
        self.icon = ui.Icon(
            "collapsed",
            self.menu,
            label=chr(0xE8F4),  # Material design icon for "remove_red_eye"
            on_val=False,
            off_val=True,
            setter=self.toggle_menu,
            label_font="pupil_icons",
        )
        self.icon.tooltip = "Eyelid Detector Settings"
        self.g_pool.iconbar.append(self.icon)

        self.menu.add(
            ui.Selector(
                "camera_position",
                self,
                label="Camera Position",
                selection=[
                    "left_below",
                    "left_side",
                    "right_below",
                    "right_side",
                ],
                setter=self.set_camera_position,
            )
        )
        self.menu.add(ui.Switch("model_enabled", self, label="Enable Model"))
        self.menu.add(ui.Switch("ear_enabled", self, label="Enable EAR"))

        self.menu.add(
            ui.Text_Input(
                "model_path_below",
                self,
                label="Model Path (Below)",
            )
        )
        self.menu.add(
            ui.Text_Input(
                "model_path_side",
                self,
                label="Model Path (Side)",
            )
        )
        self.menu.add(ui.Button("Reload Models", self.load_model))

        self.font = font.Font(self.g_pool.gui.theme.font_path, self.g_pool.gui.theme.font_size)
        self.text_texture = Named_Texture()
        self.frame_texture = Named_Texture()
        self.load_model()


    def toggle_menu(self, collapsed):
        self.g_pool.menubar.collapsed = collapsed
        for m in self.g_pool.menubar.elements:
            if m is not self.menu:
                m.collapsed = True
        self.menu.collapsed = collapsed

    def add_menu(self):
        self.menu = ui.Growing_Menu(self.menu_name, header_pos="headline")
        self.g_pool.menubar.append(self.menu)

    def set_camera_position(self, position):
        self.camera_position = position
        self.load_model() # Reload model when position changes

    def load_model(self):
        try:
            if "below" in self.camera_position:
                path = self.model_path_below
            else:
                path = self.model_path_side

            if not os.path.exists(path):
                self.log_error(f"Model file not found at {path}")
                self.model = None
                return

            self.model = AdaptiveEyeNet6Layers()
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            self.log_info(f"Successfully loaded model from {path}")
        except Exception as e:
            self.log_error(f"Failed to load model: {e}")
            self.model = None

    def recent_events(self, events):
        frame = events.get("frame")
        if frame:
            if self.model_enabled and self.model:
                # 1. Rotate the frame
                self.rotated_frame = apply_camera_rotation(
                    frame.gray, self.camera_position
                )

                # 2. Preprocess the frame for the model
                img = cv2.resize(self.rotated_frame, (192, 192))
                img_tensor = (
                    torch.from_numpy(img).float().to(self.device).unsqueeze(0).unsqueeze(0)
                )

                # 3. Run the model
                with torch.no_grad():
                    self.landmarks = self.model(img_tensor).cpu().numpy().reshape(-1, 2)

                # 4. Scale landmarks to original frame size
                h, w = self.rotated_frame.shape
                self.landmarks[:, 0] *= w / 192
                self.landmarks[:, 1] *= h / 192

                # 5. Calculate EAR if enabled
                if self.ear_enabled:
                    self.ear_value = calculate_ear(self.landmarks)
            else:
                self.rotated_frame = None
                self.landmarks = None


    def gl_display(self):
        if self.model_enabled and self.model and self.rotated_frame is not None:
            # Create a texture from the rotated frame
            self.frame_texture.update_from_ndarray(self.rotated_frame)
            self.frame_texture.draw()

            # Draw landmarks on top of the frame
            h, w = self.rotated_frame.shape
            norm_landmarks = self.landmarks.copy()
            norm_landmarks[:, 0] /= w
            norm_landmarks[:, 1] /= h

            points = [tuple(p) for p in norm_landmarks]

            draw_gl_points_norm(points, size=5, color=(0.,1.,0.,1.))

        if self.ear_enabled:
            text = f"EAR: {self.ear_value:.2f}"
            self.font.render_on_texture(text, self.text_texture)

            # Find the fps_graph and position the text next to it
            fps_graph = None
            for g in self.g_pool.graphs:
                if "FPS" in g.label:
                    fps_graph = g
                    break

            if fps_graph:
                x = fps_graph.pos[0] + 120  # Position next to FPS graph
                y = fps_graph.pos[1]
            else:
                # Fallback position
                x = 260
                y = 50

            self.text_texture.draw(x, y, (1.0, 1.0, 1.0, 1.0))

    def get_init_props(self):
        return {
            "menu_name": self.menu_name,
            "model_path_below": self.model_path_below,
            "model_path_side": self.model_path_side,
        }

    def deinit_ui(self):
        if self.icon:
            self.g_pool.iconbar.remove(self.icon)
            self.icon = None
        if self.menu:
            self.g_pool.menubar.remove(self.menu)
            self.menu = None
