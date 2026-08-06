import itertools
from contextlib import contextmanager

import napari
import numpy as np
from napari.layers import Image, Labels, Points
from napari.layers.points._points_mouse_bindings import DRAG_DIST_THRESHOLD
from napari.layers.points._points_mouse_bindings import add as napari_add_point
from napari.layers.utils._link_layers import get_linked_layers, unlink_layers
from napari.layers.utils.plane import ClippingPlane
from napari.utils.geometry import point_in_bounding_box
from qtpy import QtCore
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledRangeSlider, QLabeledSlider


@contextmanager
def silenced(slider):
    """Update a slider without it writing back to the layers

    Changing the range of a slider makes it re-emit its (stale) value, which would
    otherwise be applied to the layer whose settings we are about to restore.
    """

    was_blocked = slider.signalsBlocked()
    slider.blockSignals(True)
    try:
        yield
    finally:
        slider.blockSignals(was_blocked)


def displayed_layer_dims(layer, event) -> list[int]:
    """Map the displayed dimensions of the event onto the dimensions of the layer

    The event carries the displayed dimensions of the viewer, while the layer methods
    below index into the dimensions of the layer. Both are only the same when the layer
    has as many dimensions as the viewer.
    """

    return list(
        layer._world_to_layer_dims(
            world_dims=event.dims_displayed, ndim_world=len(event.position)
        )
    )


def plane_click_position(plane_layer, event):
    """Where the click of the event crosses the plane of the layer, in world coordinates

    Returns None if the click does not land on the plane within the layer.
    """

    dims_displayed = displayed_layer_dims(plane_layer, event)
    if len(dims_displayed) < 3:
        return None

    # the ray under the mouse, in the data coordinates of the plane layer
    position = np.asarray(
        plane_layer.world_to_data(event.position), dtype=float
    )
    direction = plane_layer._world_to_displayed_data_ray(
        np.asarray(event.view_direction), dims_displayed
    )

    intersection = plane_layer.plane.intersect_with_line(
        line_position=position[dims_displayed], line_direction=direction
    )
    if not point_in_bounding_box(
        intersection, plane_layer.extent.data[:, dims_displayed]
    ):
        return None

    # the dimensions the plane layer does not have keep the position of the event
    position[dims_displayed] = intersection
    world_position = np.asarray(event.position, dtype=float)
    world_position[-plane_layer.ndim :] = plane_layer.data_to_world(position)
    return world_position


class PlaneSliderWidget(QWidget):
    """Widget implementing sliders for 3D plane and 3D clipping plane visualization"""

    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()

        self.viewer = viewer
        self.viewer.dims.events.ndisplay.connect(self.on_ndisplay_changed)
        self.viewer.layers.selection.events.changed.connect(
            self._on_selection_changed
        )

        # snap the reported cursor position to the plane, before any other callback reads it
        self.viewer.mouse_move_callbacks.insert(0, self._snap_cursor_to_plane)
        self.viewer.mouse_drag_callbacks.insert(0, self._snap_cursor_to_plane)
        self.viewer.mouse_double_click_callbacks.insert(
            0, self._snap_cursor_to_plane
        )

        self.mode = "slice"
        self.current_layer = None

        # Add instruction label
        self.instruction_label = QLabel(
            "<i>Plane and clipping plane controls are only available in 3D mode</i>"
        )
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setVisible(self.viewer.dims.ndisplay == 2)

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

        # Create plane normal widget with buttons for the different viewing directions
        plane_labels_layout = QHBoxLayout()
        plane_labels_layout.addWidget(QLabel("Plane Normal"))
        plane_set_x_btn = QPushButton("X")
        plane_set_x_btn.clicked.connect(self._set_x_orientation)
        plane_set_y_btn = QPushButton("Y")
        plane_set_y_btn.clicked.connect(self._set_y_orientation)
        plane_set_z_btn = QPushButton("Z")
        plane_set_z_btn.clicked.connect(self._set_z_orientation)
        plane_set_oblique_btn = QPushButton("Oblique")
        plane_set_oblique_btn.clicked.connect(self._set_oblique_orientation)
        plane_labels_layout.addWidget(plane_set_x_btn)
        plane_labels_layout.addWidget(plane_set_y_btn)
        plane_labels_layout.addWidget(plane_set_z_btn)
        plane_labels_layout.addWidget(plane_set_oblique_btn)
        self.plane_labels = QWidget()
        self.plane_labels.setLayout(plane_labels_layout)

        # Single slider for plane position
        plane_layout = QVBoxLayout()
        plane_layout.addWidget(QLabel("Plane"))
        self.plane_slider = QLabeledSlider(QtCore.Qt.Horizontal)
        self.plane_slider.setSingleStep(1)
        self.plane_slider.setTickInterval(1)
        self.plane_slider.setValue(0)
        self.plane_slider.setEnabled(False)
        self.plane_slider.valueChanged.connect(self._set_plane)

        plane_layout.addWidget(self.plane_slider)

        # Snapping of the reported position and of new points to the plane
        self.snap_checkbox = QCheckBox("Snap to plane")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip(
            "Snap mouse position to the plane and place points added to a linked\n"
            "points layer on the plane"
        )

        self.snap_checkbox.toggled.connect(self._update_point_snapping)
        plane_layout.addWidget(self.snap_checkbox)

        # Thickness of the slab that linked layers without a plane (e.g. Points) are clipped to
        slab_layout = QHBoxLayout()
        slab_layout.addWidget(QLabel("Linked clipping plane thickness"))
        self.slab_thickness_box = QSpinBox()
        self.slab_thickness_box.setRange(1, 1000)
        self.slab_thickness_box.setValue(5)
        self.slab_thickness_box.valueChanged.connect(self._update_linked_slab)
        slab_layout.addWidget(self.slab_thickness_box)
        self.slab_widget = QWidget()
        self.slab_widget.setLayout(slab_layout)
        self.slab_widget.setVisible(False)
        plane_layout.addWidget(self.slab_widget)

        self.plane_widget = QWidget()
        self.plane_widget.setLayout(plane_layout)

        # Range slider for clipping planes
        clipping_plane_layout = QVBoxLayout()
        clipping_plane_layout.addWidget(QLabel("Clipping Plane"))
        self.clipping_plane_slider = QLabeledRangeSlider(QtCore.Qt.Horizontal)
        self.clipping_plane_slider.setValue((0, 1))
        self.clipping_plane_slider.valueChanged.connect(
            self._set_clipping_plane
        )
        self.clipping_plane_slider.setSingleStep(1)
        self.clipping_plane_slider.setTickInterval(1)
        self.clipping_plane_slider.setEnabled(False)

        clipping_plane_layout.addWidget(self.clipping_plane_slider)
        self.clipping_plane_widget = QWidget()
        self.clipping_plane_widget.setLayout(clipping_plane_layout)

        # Combine buttons and sliders
        plane_layout = QVBoxLayout()
        plane_layout.addWidget(self.plane_labels)
        plane_layout.addWidget(self.plane_widget)
        plane_layout.addWidget(self.clipping_plane_widget)

        # Assemble main layout
        view_mode_widget_layout = QVBoxLayout()
        view_mode_widget_layout.addWidget(self.instruction_label)
        view_mode_widget_layout.addLayout(button_layout)
        view_mode_widget_layout.addLayout(plane_layout)

        # Hide plane controls until needed
        self.plane_labels.setVisible(False)
        self.plane_widget.setVisible(False)
        self.clipping_plane_widget.setVisible(False)

        self.setLayout(view_mode_widget_layout)

    def _on_selection_changed(self) -> None:
        """Update the active layer"""

        if (
            len(self.viewer.layers.selection) == 1
        ):  # Only consider single layer selection
            selected_layer = self.viewer.layers.selection.active
            if isinstance(selected_layer, Labels | Image | Points):
                self.current_layer = selected_layer
            else:
                self.current_layer = None
                return

            self._ensure_clipping_planes(self.current_layer)

            # a layer without a plane of its own can still drive the plane of a
            # layer it is linked to
            plane_layer = self._plane_layer()
            self.plane_btn.setVisible(plane_layer is not None)
            if plane_layer is not None:
                plane_layer.events.plane.connect(self._update_plane_slider)
                plane_layer.events.depiction.connect(self._update_mode)

            self._update_slab_visibility()
            self._update_point_snapping()
            self._release_mouse_locks()

            if self.viewer.dims.ndisplay == 3:
                self._update_view_mode()

    def _ensure_clipping_planes(self, layer) -> None:
        """Give a layer the pair of clipping planes this widget operates on"""

        if len(layer.experimental_clipping_planes) != 0:
            return

        if hasattr(layer, "plane"):
            normal, position = layer.plane.normal, layer.plane.position
        else:
            # layers without a plane follow the orientation shown in the widget
            normal, position = self._plane_normal(), (0, 0, 0)

        layer.experimental_clipping_planes.append(
            ClippingPlane(normal=normal, position=position, enabled=False)
        )
        layer.experimental_clipping_planes.append(
            ClippingPlane(
                normal=[-n for n in normal], position=position, enabled=False
            )
        )

    def _linked_layers(self) -> list:
        """The layers the user linked the current layer to"""

        if self.current_layer is None:
            return []
        return list(get_linked_layers(self.current_layer))

    def _target_layers(self) -> list:
        """The layers to apply clipping plane changes to: the current layer and the layers it is linked to"""

        if self.current_layer is None:
            return []

        layers = [self.current_layer, *self._linked_layers()]
        for layer in layers:
            self._ensure_clipping_planes(layer)
        return layers

    @staticmethod
    def _refresh(layer) -> None:
        """Redraw a layer whose visual does not update itself when the clipping planes change

        Image and Labels layers redraw on their own, but the visuals using the vispy
        clipping planes filter (Points, Shapes, Surface, ...) only pick the new planes
        up on the next canvas draw.

        Only the visual is asked to update. Refreshing the layer would slice it again,
        which rebuilds the arrays the status bar thread reads while it is reading them,
        even though the clipping planes leave the data untouched.
        """

        if not isinstance(layer, Image | Labels):
            layer.events.set_data()

    def _update_linked_layers(self) -> None:
        """Let the linked layers follow the plane of the current layer"""

        self._sync_linked_planes()
        self._update_linked_slab()
        self._update_point_snapping()

    def _active_plane_layer(self):
        """The layer whose plane the cursor should snap to, if any"""

        if not self.snap_checkbox.isChecked():
            return None
        if self.viewer.dims.ndisplay != 3:
            return None

        layer = self.viewer.layers.selection.active
        if layer is None:
            return None
        if (
            getattr(layer, "depiction", None) == "plane"
            and layer.visible
            and layer.ndim >= 3
        ):
            return layer
        return self._plane_layer_for(layer)

    def _snap_cursor_to_plane(self, viewer, event) -> None:
        """Report the position on the plane while a plane is shown, instead of where the
        click ray enters the data.
        """

        plane_layer = self._active_plane_layer()
        if plane_layer is None:
            return

        position = plane_click_position(plane_layer, event)
        if position is not None:
            viewer.cursor.position = tuple(position)

    def _release_mouse_locks(self) -> None:
        """Keep the mode of one layer from locking the camera on the layers linked to it

        napari links `mouse_pan` and `mouse_zoom` between linked layers, but it does not
        link `mode`. A points layer in select or transform mode therefore switches
        panning off on the image it is linked to, where nothing ever switches it back
        on: that image keeps its own pan and zoom mode, and unlinking the layers leaves
        the lock behind.
        """

        layers = self._target_layers()
        if len(layers) > 1:
            unlink_layers(layers, ("mouse_pan", "mouse_zoom"))

        # a layer that is in its own pan and zoom mode should not be left blocked by
        # the mode of another layer, however it got there
        for layer in self.viewer.layers:
            if str(getattr(layer, "mode", "")) == "pan_zoom" and not (
                layer.mouse_pan and layer.mouse_zoom
            ):
                layer.mouse_pan = True
                layer.mouse_zoom = True

    def _plane_layer(self):
        """The layer whose plane the plane controls act on

        A layer without a plane of its own, such as a points layer, falls back to a
        layer it is linked to, so that the plane can be moved while the points layer
        stays selected.
        """

        if self.current_layer is None:
            return None
        if hasattr(self.current_layer, "plane"):
            return self.current_layer

        for layer in self._linked_layers():
            if hasattr(layer, "plane") and layer.visible and layer.ndim >= 3:
                return layer
        return None

    def _plane_layer_for(self, layer):
        """The linked layer whose plane new points of this layer should land on"""

        if not self.snap_checkbox.isChecked():
            return None

        for other in get_linked_layers(layer):
            if (
                getattr(other, "depiction", None) == "plane"
                and other.visible
                and other.ndim >= 3
            ):
                return other
        return None

    def _update_point_snapping(self) -> None:
        """Snap points added to a linked points layer onto the plane of the current layer"""

        for layer in self._target_layers():
            if not isinstance(layer, Points):
                continue
            layer.events.mode.connect(self._on_point_mode_changed)
            self._update_point_add_callback(layer)

    def _on_point_mode_changed(self, event) -> None:
        """napari restores its own callbacks whenever the mode of a layer changes"""

        self._update_point_add_callback(event.source)
        self._release_mouse_locks()

    def _update_point_add_callback(self, layer) -> None:
        """Replace the callback that adds points while snapping applies"""

        callbacks = layer.mouse_drag_callbacks
        snapping = (
            str(layer.mode) == "add"
            and self._plane_layer_for(layer) is not None
        )

        if snapping:
            if napari_add_point in callbacks:
                callbacks.remove(napari_add_point)
            if self._add_point_on_plane not in callbacks:
                callbacks.append(self._add_point_on_plane)
        elif self._add_point_on_plane in callbacks:
            callbacks.remove(self._add_point_on_plane)
            if str(layer.mode) == "add" and napari_add_point not in callbacks:
                callbacks.append(napari_add_point)

    def _add_point_on_plane(self, layer, event):
        """Add a point where the click crosses the plane of the linked layer

        Same click and drag handling as the napari callback, but the point is placed
        on the plane. A click that does not land on the plane adds no point, rather
        than one somewhere along the line of sight.
        """

        plane_layer = self._plane_layer_for(layer)
        if plane_layer is None:
            yield from napari_add_point(layer, event)
            return

        start_pos = event.pos
        yield

        while event.type == "mouse_move":
            if np.linalg.norm(start_pos - event.pos) < DRAG_DIST_THRESHOLD:
                # prevent vispy from moving the canvas if we're below threshold
                event.handled = True
            yield

        if np.linalg.norm(start_pos - event.pos) >= DRAG_DIST_THRESHOLD:
            return

        position = plane_click_position(plane_layer, event)
        if position is not None:
            layer.add(layer.world_to_data(position))

    def _sync_linked_planes(self) -> None:
        """Copy the plane the widget acts on to the other linked layers that have one

        napari only links attributes that all linked layers have in common, so an image
        and a labels layer stop syncing their plane as soon as a layer without a plane
        (e.g. Points) joins the same link group.
        """

        plane_layer = self._plane_layer()
        if plane_layer is None:
            return

        plane = plane_layer.plane
        for layer in self._linked_layers():
            if layer is plane_layer or not hasattr(layer, "plane"):
                continue

            # only the layer the widget acts on drives the sliders
            layer.events.plane.disconnect(self._update_plane_slider)
            layer.plane.normal = plane.normal
            layer.plane.position = plane.position

    def _update_slab_visibility(self) -> None:
        """Only offer a slab thickness when a layer in the group has no plane of its own"""

        self.slab_widget.setVisible(
            self._plane_layer() is not None
            and any(
                not hasattr(layer, "plane") for layer in self._target_layers()
            )
        )

    def _update_linked_slab(self) -> None:
        """Clip the layers in the group that have no plane to a slab around the plane

        This lets layers such as Points mimic plane mode: they only show the points
        within `slab_thickness_box` slices of the plane of the image or labels layer
        they are linked to.
        """

        self._update_slab_visibility()

        plane_layer = self._plane_layer()
        if self.mode != "plane" or plane_layer is None:
            return

        normal = np.array(plane_layer.plane.normal)
        center = np.dot(plane_layer.plane.position, normal) / np.dot(
            normal, normal
        )
        half_thickness = self.slab_thickness_box.value() / 2

        for layer in self._target_layers():
            if hasattr(layer, "plane"):
                continue  # image-like layers show the plane themselves

            layer.experimental_clipping_planes[0].normal = normal
            layer.experimental_clipping_planes[1].normal = tuple(
                -n for n in normal
            )
            layer.experimental_clipping_planes[0].position = tuple(
                (center - half_thickness) * normal
            )
            layer.experimental_clipping_planes[1].position = tuple(
                (center + half_thickness) * normal
            )
            for clip_plane in layer.experimental_clipping_planes:
                clip_plane.enabled = True
            self._refresh(layer)

    def _plane_normal(self) -> np.ndarray:
        """Normal of the plane, or of the first clipping plane if there is no plane"""

        plane_layer = self._plane_layer()
        if plane_layer is not None:
            return np.array(plane_layer.plane.normal)
        if len(self.current_layer.experimental_clipping_planes) > 0:
            return np.array(
                self.current_layer.experimental_clipping_planes[0].normal
            )
        return np.array([1.0, 0.0, 0.0])

    def _layer_bounds(self):
        """Lower and upper corner of the bounding box of the current layer and its linked layers

        Linked layers share the sliders, so the range has to cover all of them. A layer
        without data has no extent of its own (an empty points layer reports NaN) and is
        left out, which is why the range of a points layer follows the image it is
        linked to instead of the points it happens to hold.
        """

        extents = [
            layer.extent.data[:, -3:]
            for layer in self._target_layers()
            if layer.extent.data.shape[1] >= 3
            and np.all(np.isfinite(layer.extent.data[:, -3:]))
        ]
        if not extents:
            return np.zeros(3), np.ones(3)

        extents = np.array(extents)
        return extents[:, 0].min(axis=0), extents[:, 1].max(axis=0)

    def _update_view_mode(self) -> None:
        """Restore the widget state from the depiction and clipping planes of the group"""

        plane_layer = self._plane_layer()
        if plane_layer is not None and plane_layer.depiction == "plane":
            self._set_plane_mode()
        elif self.current_layer.experimental_clipping_planes[0].enabled:
            self._set_clipping_plane_mode()
        else:
            self._set_volume_mode()

    def compute_plane_range(self):
        """Compute the range of the plane and clipping plane sliders"""

        normal = self._plane_normal()
        mins, maxs = self._layer_bounds()

        # Define the corners of the 3D bounding box
        corners = np.array(
            list(itertools.product(*zip(mins, maxs, strict=True)))
        )

        # Project the corners onto the normal vector
        projections = np.dot(corners, normal)

        # The range of possible positions is given by the min and max projections
        min_position = np.min(projections)
        max_position = np.max(projections)

        return min_position, max_position

    def _set_axis_orientation(self, normal, axis: int) -> None:
        """Align the plane with the given axis aligned normal"""

        if self.current_layer is None:
            return

        plane_layer = self._plane_layer()
        if plane_layer is not None:
            plane_layer.plane.normal = normal

        axis_max = int(self._layer_bounds()[1][axis])
        with silenced(self.plane_slider):
            self.plane_slider.setMinimum(0)
            self.plane_slider.setMaximum(axis_max)
        self.plane_slider.setValue(int(axis_max / 2))

        self._set_clipping_plane_normals(normal)

        with silenced(self.clipping_plane_slider):
            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(axis_max)
        self.clipping_plane_slider.setValue(
            (int(axis_max / 3), int(axis_max / 1.5))
        )

        self._update_linked_layers()

    def _set_clipping_plane_normals(self, normal) -> None:
        """Orient the clipping planes of the current layer and of the layers linked to it"""

        for layer in self._target_layers():
            layer.experimental_clipping_planes[0].normal = normal
            layer.experimental_clipping_planes[1].normal = tuple(
                -n for n in normal
            )
            self._refresh(layer)

    def _set_x_orientation(self):
        """Align the plane to slide in the yz plane"""

        self._set_axis_orientation((0, 0, 1), -1)

    def _set_y_orientation(self):
        """Align the plane to slide in the xz plane"""

        self._set_axis_orientation((0, 1, 0), -2)

    def _set_z_orientation(self):
        """Align the plane to slide in the yx plane"""

        self._set_axis_orientation((1, 0, 0), -3)

    def _set_oblique_orientation(self) -> None:
        """Orient plane normal along the viewing direction"""

        if self.current_layer is not None:
            plane_layer = self._plane_layer() or self.current_layer
            normal = plane_layer._world_to_displayed_data_ray(
                self.viewer.camera.view_direction, [-3, -2, -1]
            )
            if hasattr(plane_layer, "plane"):
                plane_layer.plane.normal = normal

            self._set_clipping_plane_normals(normal)

            min_range, max_range = self.compute_plane_range()
            with silenced(self.plane_slider):
                self.plane_slider.setMinimum(int(min_range))
                self.plane_slider.setMaximum(int(max_range))
            self.plane_slider.setValue(int(max_range / 2))

            with silenced(self.clipping_plane_slider):
                self.clipping_plane_slider.setMinimum(int(min_range))
                self.clipping_plane_slider.setMaximum(int(max_range))
            self.clipping_plane_slider.setValue(
                (int(max_range / 3), int(max_range / 1.5))
            )

            self._update_linked_layers()

    def _update_mode(self):
        """Update the mode in case the user manually changes the depiction (for synchronization purposes only)"""

        plane_layer = self._plane_layer()
        if plane_layer is None:
            return

        plane_layer.events.depiction.disconnect(self._update_mode)
        if plane_layer.depiction == "volume":
            self._set_volume_mode()
        elif plane_layer.depiction == "plane":
            self._set_plane_mode()
        plane_layer.events.depiction.connect(self._update_mode)

    def _set_depiction(self, depiction: str) -> None:
        """Set the depiction of the layers in the group that have one, without triggering the sync callback"""

        plane_layer = self._plane_layer()
        if plane_layer is None:
            # nothing in the group shows a plane, so there is no depiction to set
            return

        for layer in self._target_layers():
            if not hasattr(layer, "depiction"):
                continue
            # only the layer the widget acts on drives the sliders
            layer.events.depiction.disconnect(self._update_mode)
            layer.depiction = depiction

        plane_layer.events.depiction.connect(self._update_mode)
        self._update_point_snapping()

    def _update_clipping_plane_slider(self):
        """Updates the values of the clipping plane slider when switching between different layers"""

        new_position = np.array(
            self.current_layer.experimental_clipping_planes[0].position
        )
        plane_normal = np.array(
            self.current_layer.experimental_clipping_planes[0].normal
        )
        slider_value1 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        new_position = np.array(
            self.current_layer.experimental_clipping_planes[1].position
        )
        plane_normal = np.array(
            self.current_layer.experimental_clipping_planes[0].normal
        )
        slider_value2 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        with silenced(self.clipping_plane_slider):
            self.clipping_plane_slider.setValue(
                (int(slider_value1), int(slider_value2))
            )

    def _update_plane_slider(self):
        """Updates the value of the plane slider when the user used the shift+drag method to shift the plane or when switching between different layers"""

        plane_layer = self._plane_layer()
        if plane_layer is None:
            return

        new_position = np.array(plane_layer.plane.position)
        plane_normal = np.array(plane_layer.plane.normal)
        slider_value = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )
        with silenced(self.plane_slider):
            self.plane_slider.setValue(int(slider_value))

        self._update_linked_layers()

    def _set_plane_mode(self) -> None:
        """Activate the plane mode on the current layer"""

        if self.current_layer is not None:
            self.mode = "plane"
            self._set_depiction("plane")

            self.plane_btn.setEnabled(False)
            self.clipping_plane_btn.setEnabled(True)
            self.volume_btn.setEnabled(True)

            self.plane_slider.setEnabled(True)
            self.clipping_plane_slider.setEnabled(False)

            self._enable_clipping_planes(False)

            min_range, max_range = self.compute_plane_range()
            with silenced(self.plane_slider):
                self.plane_slider.setMinimum(int(min_range))
                self.plane_slider.setMaximum(int(max_range))

            # restore the plane position that is stored on the layer
            self._update_plane_slider()
            if self.plane_slider.value() == 0:
                self.plane_slider.setValue(int(max_range / 2))

            self.plane_labels.setVisible(True)
            self.plane_widget.setVisible(True)
            self.clipping_plane_widget.setVisible(False)

    def _set_clipping_plane_mode(self) -> None:
        """Activate the clipping plane mode on the current layer"""

        if self.current_layer is not None:
            self.mode = "clipping_plane"
            self._set_depiction("volume")

            self.plane_btn.setEnabled(True)
            self.clipping_plane_btn.setEnabled(False)
            self.volume_btn.setEnabled(True)

            self.plane_slider.setEnabled(False)
            self.clipping_plane_slider.setEnabled(True)

            self._enable_clipping_planes(True)

            min_range, max_range = self.compute_plane_range()
            with silenced(self.clipping_plane_slider):
                self.clipping_plane_slider.setMinimum(int(min_range))
                self.clipping_plane_slider.setMaximum(int(max_range))

            # restore the clipping plane positions that are stored on the layer
            self._update_clipping_plane_slider()
            lower, upper = self.clipping_plane_slider.value()
            if lower >= upper:  # nothing meaningful stored yet
                self.clipping_plane_slider.setValue(
                    (int(max_range / 3), int(max_range / 1.5))
                )
            else:
                # re-apply, in case a layer was linked since the positions were set
                self._set_clipping_plane()

            self.plane_labels.setVisible(True)
            self.plane_widget.setVisible(False)
            self.clipping_plane_widget.setVisible(True)

    def _set_volume_mode(self) -> None:
        """Deactive plane viewing and go back to default volume viewing"""

        if self.current_layer is not None:
            self.mode = "volume"

            self.plane_btn.setEnabled(True)
            self.clipping_plane_btn.setEnabled(True)
            self.volume_btn.setEnabled(False)

            self.plane_slider.setEnabled(False)
            self.clipping_plane_slider.setEnabled(False)

            self._set_depiction("volume")
            self._enable_clipping_planes(False)

            self.plane_labels.setVisible(False)
            self.plane_widget.setVisible(False)
            self.clipping_plane_widget.setVisible(False)

    def on_ndisplay_changed(self) -> None:
        """Update the buttons depending on the display mode of the viewer. Buttons and sliders should only be active in 3D mode"""

        if self.viewer.dims.ndisplay == 2:
            self.plane_btn.setEnabled(False)
            self.volume_btn.setEnabled(False)
            self.clipping_plane_btn.setEnabled(False)
            self.plane_slider.setEnabled(False)
            self.clipping_plane_slider.setEnabled(False)
            self.mode = "slice"
            self.plane_labels.setVisible(False)
            self.plane_widget.setVisible(False)
            self.clipping_plane_widget.setVisible(False)
            self.instruction_label.setVisible(True)

        elif self.current_layer is not None:
            self._update_view_mode()
            self.instruction_label.setVisible(False)

    def _enable_clipping_planes(self, enabled: bool) -> None:
        """Enable or disable the clipping planes of the current layer and of the layers linked to it"""

        for layer in self._target_layers():
            if enabled and getattr(layer, "depiction", None) == "plane":
                continue  # this layer is showing a plane, it should not be clipped
            for clip_plane in layer.experimental_clipping_planes:
                clip_plane.enabled = enabled
            self._refresh(layer)

    def _dims_steps(self) -> np.ndarray:
        """Step size of the last three dimensions of the viewer

        Falls back to a step of one for a viewer that does not have three dimensions
        yet, for instance when the only layer is an empty points layer.
        """

        ranges = self.viewer.dims.range
        if len(ranges) < 3:
            return np.ones(3)
        return np.array([axis.step for axis in ranges[-3:]])

    def _set_clipping_plane(self) -> None:
        """Adjust the range of the clipping plane"""

        if self.current_layer is not None:
            plane_normal = np.array(
                self.current_layer.experimental_clipping_planes[0].normal
            )
            slider_value = self.clipping_plane_slider.value()
            steps = self._dims_steps()
            new_position_1 = (
                np.array([0, 0, 0]) + slider_value[0] * plane_normal
            )
            new_position_1 = (
                int(new_position_1[0] * steps[0]),
                (new_position_1[1] * steps[1]),
                int(new_position_1[2] * steps[2]),
            )
            new_position_2 = (
                np.array([0, 0, 0]) + slider_value[1] * plane_normal
            )
            new_position_2 = (
                int(new_position_2[0] * steps[0]),
                (new_position_2[1] * steps[1]),
                int(new_position_2[2] * steps[2]),
            )

            for layer in self._target_layers():
                layer.experimental_clipping_planes[0].position = new_position_1
                layer.experimental_clipping_planes[1].position = new_position_2
                self._refresh(layer)

    def _set_plane(self) -> None:
        """Move the plane to a new location"""

        plane_layer = self._plane_layer()
        if plane_layer is not None:
            plane_normal = np.array(plane_layer.plane.normal)
            slider_value = self.plane_slider.value()
            new_position = np.array([0, 0, 0]) + slider_value * plane_normal
            plane_layer.plane.position = tuple(new_position)
