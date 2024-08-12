import napari.layers
import numpy as np
from napari.layers.utils.plane import ClippingPlane
from qtpy import QtCore
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledRangeSlider, QLabeledSlider

from .layer_dropdown import LayerDropdown


class PlaneSliderWidget(QWidget):
    """Widget implementing sliders for 3D plane and 3D clipping plane visualization"""

    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()

        self.viewer = viewer
        self.viewer.dims.events.ndisplay.connect(self.on_ndisplay_changed)
        self.mode = "slice"
        self.current_layer = None

        # Add a dropdown menu to select a layer to add a plane view for
        self.layer_dropdown = LayerDropdown(
            self.viewer, (napari.layers.Image, napari.layers.Labels)
        )
        self.layer_dropdown.layer_changed.connect(self._update_layer)

        # Add buttons to switch between plane and volume mode
        button_layout = QVBoxLayout()
        plane_volume_layout = QHBoxLayout()
        self.plane_btn = QPushButton("Plane")
        self.plane_btn.setEnabled(False)
        self.plane_btn.clicked.connect(self._set_plane_mode)
        self.clipping_plane_btn = QPushButton("Clipping Plane")
        self.clipping_plane_btn.setEnabled(False)
        self.clipping_plane_btn.clicked.connect(self._set_clipping_plane_mode)
        self.volume_btn = QPushButton("Volume")
        self.volume_btn.setEnabled(False)
        self.volume_btn.clicked.connect(self._set_volume_mode)
        plane_volume_layout.addWidget(self.plane_btn)
        plane_volume_layout.addWidget(self.clipping_plane_btn)
        plane_volume_layout.addWidget(self.volume_btn)
        button_layout.addLayout(plane_volume_layout)

        # Add buttons for the different viewing directions
        plane_labels = QHBoxLayout()
        plane_set_x_btn = QPushButton("X")
        plane_set_x_btn.clicked.connect(self._set_x_orientation)
        plane_set_y_btn = QPushButton("Y")
        plane_set_y_btn.clicked.connect(self._set_y_orientation)
        plane_set_z_btn = QPushButton("Z")
        plane_set_z_btn.clicked.connect(self._set_z_orientation)
        plane_set_oblique_btn = QPushButton("Oblique")
        plane_set_oblique_btn.clicked.connect(self._set_oblique_orientation)
        plane_labels.addWidget(plane_set_x_btn)
        plane_labels.addWidget(plane_set_y_btn)
        plane_labels.addWidget(plane_set_z_btn)
        plane_labels.addWidget(plane_set_oblique_btn)

        # Add plane sliders for viewing in 3D
        self.plane_slider = QLabeledSlider(QtCore.Qt.Horizontal)
        self.plane_slider.setSingleStep(1)
        self.plane_slider.setTickInterval(1)
        self.plane_slider.setValue(0)
        self.plane_slider.setEnabled(False)
        self.plane_slider.valueChanged.connect(self._set_plane)

        self.clipping_plane_slider = QLabeledRangeSlider(QtCore.Qt.Horizontal)
        self.clipping_plane_slider.setValue((0, 1))
        self.clipping_plane_slider.valueChanged.connect(
            self._set_clipping_plane
        )
        self.clipping_plane_slider.setSingleStep(1)
        self.clipping_plane_slider.setTickInterval(1)
        self.clipping_plane_slider.setEnabled(False)

        # Combine buttons and sliders
        plane_layout = QVBoxLayout()
        plane_layout.addWidget(QLabel("Plane Normal"))
        plane_layout.addLayout(plane_labels)
        plane_layout.addWidget(QLabel("Plane"))
        plane_layout.addWidget(self.plane_slider)
        plane_layout.addWidget(QLabel("Clipping Plane"))
        plane_layout.addWidget(self.clipping_plane_slider)

        # Assemble main layout
        view_mode_widget_layout = QVBoxLayout()
        view_mode_widget_layout.addWidget(self.layer_dropdown)
        view_mode_widget_layout.addLayout(button_layout)
        view_mode_widget_layout.addLayout(plane_layout)

        self.setLayout(view_mode_widget_layout)

    def compute_plane_range(self):
        """Compute the range of the plane and clipping plane sliders"""

        normal = np.array(self.current_layer.plane.normal)
        Lx, Ly, Lz = self.current_layer.data.shape[-3:]

        # Define the corners of the 3D image bounding box
        corners = np.array(
            [
                [0, 0, 0],
                [Lx, 0, 0],
                [0, Ly, 0],
                [0, 0, Lz],
                [Lx, Ly, 0],
                [Lx, 0, Lz],
                [0, Ly, Lz],
                [Lx, Ly, Lz],
            ]
        )

        # Project the corners onto the normal vector
        projections = np.dot(corners, normal)

        # The range of possible positions is given by the min and max projections
        min_position = np.min(projections)
        max_position = np.max(projections)

        return min_position, max_position

    def _set_x_orientation(self):
        """Align the plane to slide in the yz plane"""

        if self.current_layer is not None:
            self.current_layer.plane.normal = (0, 0, 1)
            self.plane_slider.setMinimum(0)
            self.plane_slider.setMaximum(self.current_layer.data.shape[-1])
            self.plane_slider.setValue(
                int(self.current_layer.data.shape[-1] / 2)
            )

            self.current_layer.experimental_clipping_planes[0].normal = (
                0,
                0,
                1,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                0,
                0,
                -1,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(
                self.current_layer.data.shape[-1]
            )
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-1] / 3),
                    int(self.current_layer.data.shape[-1] / 1.5),
                )
            )

    def _set_y_orientation(self):
        """Align the plane to slide in the xz plane"""

        if self.current_layer is not None:

            self.current_layer.plane.normal = (0, 1, 0)
            self.plane_slider.setMinimum(0)
            self.plane_slider.setMaximum(self.current_layer.data.shape[-2])
            self.plane_slider.setValue(
                int(self.current_layer.data.shape[-2] / 2)
            )

            self.current_layer.experimental_clipping_planes[0].normal = (
                0,
                1,
                0,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                0,
                -1,
                0,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(
                self.current_layer.data.shape[-2]
            )
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-2] / 3),
                    int(self.current_layer.data.shape[-2] / 1.5),
                )
            )

    def _set_z_orientation(self):
        """Align the plane to slide in the yx plane"""

        if self.current_layer is not None:

            self.current_layer.plane.normal = (1, 0, 0)
            self.plane_slider.setMinimum(0)
            self.plane_slider.setMaximum(self.current_layer.data.shape[-3])
            self.plane_slider.setValue(
                int(self.current_layer.data.shape[-3] / 2)
            )

            self.current_layer.experimental_clipping_planes[0].normal = (
                1,
                0,
                0,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                -1,
                0,
                0,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(
                self.current_layer.data.shape[-3]
            )
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-3] / 3),
                    int(self.current_layer.data.shape[-3] / 1.5),
                )
            )

    def _set_oblique_orientation(self) -> None:
        """Orient plane normal along the viewing direction"""

        if self.current_layer is not None:

            self.current_layer.plane.normal = (
                self.current_layer._world_to_displayed_data_ray(
                    self.viewer.camera.view_direction, [-3, -2, -1]
                )
            )
            min_range, max_range = self.compute_plane_range()
            self.plane_slider.setMinimum(int(min_range))
            self.plane_slider.setMaximum(int(max_range))
            self.plane_slider.setValue(int(max_range / 2))

            self.current_layer.experimental_clipping_planes[0].normal = (
                self.current_layer.plane.normal
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                -self.current_layer.plane.normal[-3],
                -self.current_layer.plane.normal[-2],
                -self.current_layer.plane.normal[-1],
            )

            self.clipping_plane_slider.setMinimum(min_range)
            self.clipping_plane_slider.setMaximum(max_range)
            self.clipping_plane_slider.setValue(
                (int(max_range / 3), int(max_range / 1.5))
            )

    def _update_layer(self, selected_layer) -> None:
        """Update the layer to which the plane viewing is applied"""

        if selected_layer == "":
            self.current_layer = None
        else:
            self.current_layer = self.viewer.layers[selected_layer]
            self.layer_dropdown.setCurrentText(selected_layer)
            if len(self.current_layer.experimental_clipping_planes) == 0:
                plane = self.current_layer.plane
                self.current_layer.experimental_clipping_planes.append(
                    ClippingPlane(
                        normal=plane.normal,
                        position=plane.position,
                        enabled=False,
                    )
                )
                self.current_layer.experimental_clipping_planes.append(
                    ClippingPlane(
                        normal=[-n for n in plane.normal],
                        position=plane.position,
                        enabled=False,
                    )
                )
                self.current_layer.events.plane.connect(
                    self._update_plane_slider
                )
                self.current_layer.events.depiction.connect(self._update_mode)

            if self.viewer.dims.ndisplay == 3:
                if self.current_layer.depiction == "plane":
                    self._set_plane_mode()
                    self._update_plane_slider()
                elif (
                    self.current_layer.depiction == "volume"
                    and self.current_layer.experimental_clipping_planes[
                        0
                    ].enabled
                ):
                    self._set_clipping_plane_mode()
                    self._update_clipping_plane_slider()
                else:
                    self._set_volume_mode()

    def _update_mode(self):
        """Update the mode in case the user manually changes the depiction (for synchronization purposes only)"""

        self.current_layer.events.depiction.disconnect(self._update_mode)
        if self.current_layer.depiction == "volume":
            self._set_volume_mode()
        elif self.current_layer.depiction == "plane":
            self._set_plane_mode()
        self.current_layer.events.depiction.disconnect(self._update_mode)

    def _update_clipping_plane_slider(self): 
        """Updates the values of the clipping plane slider when switching between different layers"""

        new_position = np.array(self.current_layer.experimental_clipping_planes[0].position)
        plane_normal = np.array(self.current_layer.experimental_clipping_planes[0].normal)
        slider_value1 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        new_position = np.array(self.current_layer.experimental_clipping_planes[1].position)
        plane_normal = np.array(self.current_layer.experimental_clipping_planes[0].normal)
        slider_value2 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        self.clipping_plane_slider.valueChanged.disconnect(self._set_clipping_plane)
        self.clipping_plane_slider.setValue((int(slider_value1), int(slider_value2)))
        self.clipping_plane_slider.valueChanged.connect(self._set_clipping_plane)

    def _update_plane_slider(self):
        """Updates the value of the plane slider when the user used the shift+drag method to shift the plane or when switching between different layers"""

        new_position = np.array(self.current_layer.plane.position)
        plane_normal = np.array(self.current_layer.plane.normal)
        slider_value = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )
        self.plane_slider.valueChanged.disconnect(self._set_plane)
        self.plane_slider.setValue(int(slider_value))
        self.plane_slider.valueChanged.connect(self._set_plane)

    def _set_plane_mode(self) -> None:
        """Activate the plane mode on the current layer"""

        self.mode = "plane"
        self.current_layer.events.depiction.disconnect(self._update_mode)
        self.current_layer.depiction = "plane"
        self.current_layer.events.depiction.connect(self._update_mode)

        self.plane_btn.setEnabled(False)
        self.clipping_plane_btn.setEnabled(True)
        self.volume_btn.setEnabled(True)

        self.plane_slider.setEnabled(True)
        self.clipping_plane_slider.setEnabled(False)

        for clip_plane in self.current_layer.experimental_clipping_planes:
            clip_plane.enabled = False

        max_range = self.compute_plane_range()[1]
        self.plane_slider.setMaximum(int(max_range))
        if self.plane_slider.value() == 0:
            self.plane_slider.setValue(int(max_range / 2))

    def _set_clipping_plane_mode(self) -> None:
        """Activate the clipping plane mode on the current layer"""

        self.mode = "clipping_plane"
        self.current_layer.events.depiction.disconnect(self._update_mode)
        self.current_layer.depiction = "volume"
        self.current_layer.events.depiction.connect(self._update_mode)

        self.plane_btn.setEnabled(True)
        self.clipping_plane_btn.setEnabled(False)
        self.volume_btn.setEnabled(True)

        self.plane_slider.setEnabled(False)
        self.clipping_plane_slider.setEnabled(True)

        for clip_plane in self.current_layer.experimental_clipping_planes:
            clip_plane.enabled = True

        max_range = self.compute_plane_range()[1]
        self.clipping_plane_slider.setMaximum(max_range)
        if self.clipping_plane_slider.value()[0] == 0:
            self.clipping_plane_slider.setValue(
                (int(max_range / 3), int(max_range / 1.5))
            )

    def _set_volume_mode(self) -> None:
        """Deactive plane viewing and go back to default volume viewing"""

        self.mode = "volume"

        self.plane_btn.setEnabled(True)
        self.clipping_plane_btn.setEnabled(True)
        self.volume_btn.setEnabled(False)

        self.plane_slider.setEnabled(False)
        self.clipping_plane_slider.setEnabled(False)

        self.current_layer.events.depiction.disconnect(self._update_mode)
        self.current_layer.depiction = "volume"
        self.current_layer.events.depiction.connect(self._update_mode)
        for clip_plane in self.current_layer.experimental_clipping_planes:
            clip_plane.enabled = False

    def on_ndisplay_changed(self) -> None:
        """Update the buttons depending on the display mode of the viewer. Buttons and sliders should only be active in 3D mode"""

        if self.viewer.dims.ndisplay == 2:
            self.plane_btn.setEnabled(False)
            self.volume_btn.setEnabled(False)
            self.clipping_plane_btn.setEnabled(False)
            self.plane_slider.setEnabled(False)
            self.clipping_plane_slider.setEnabled(False)
            self.mode = "slice"

        elif self.current_layer is not None:
            if self.current_layer.depiction == "plane":
                self._set_plane_mode()
                self._update_plane_slider()
            elif (
                self.current_layer.depiction == "volume"
                and self.current_layer.experimental_clipping_planes[0].enabled
            ):
                self._set_clipping_plane_mode()
                self._update_clipping_plane_slider()
            else:
                self._set_volume_mode()

    def _set_clipping_plane(self) -> None:
        """Adjust the range of the clipping plane"""

        plane_normal = np.array(
            self.current_layer.experimental_clipping_planes[0].normal
        )
        slider_value = self.clipping_plane_slider.value()
        new_position_1 = np.array([0, 0, 0]) + slider_value[0] * plane_normal
        new_position_1 = (
            int(new_position_1[0] * self.viewer.dims.range[-3].step),
            (new_position_1[1] * self.viewer.dims.range[-2].step),
            int(new_position_1[2] * self.viewer.dims.range[-1].step),
        )
        self.current_layer.experimental_clipping_planes[0].position = (
            new_position_1
        )
        new_position_2 = np.array([0, 0, 0]) + slider_value[1] * plane_normal
        new_position_2 = (
            int(new_position_2[0] * self.viewer.dims.range[-3].step),
            (new_position_2[1] * self.viewer.dims.range[-2].step),
            int(new_position_2[2] * self.viewer.dims.range[-1].step),
        )

        self.current_layer.experimental_clipping_planes[1].position = (
            new_position_2
        )

    def _set_plane(self) -> None:
        """Move the plane to a new location"""

        plane_normal = np.array(self.current_layer.plane.normal)
        slider_value = self.plane_slider.value()
        new_position = np.array([0, 0, 0]) + slider_value * plane_normal
        self.current_layer.plane.position = tuple(new_position)
