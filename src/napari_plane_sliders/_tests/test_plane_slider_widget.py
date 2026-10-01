from types import SimpleNamespace

import numpy as np
import pytest
from napari.experimental import link_layers, unlink_layers
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_plane_sliders import PlaneSliderWidget


@pytest.fixture
def viewer(make_napari_viewer):
    """A viewer in 3D mode, the only mode in which the widget is active"""

    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    return viewer


@pytest.fixture
def widget(viewer):
    return PlaneSliderWidget(viewer)


@pytest.fixture
def image(viewer):
    return viewer.add_image(np.random.random((30, 40, 50)), name="image")


@pytest.fixture
def points(viewer):
    return viewer.add_points(np.random.random((20, 3)) * 30, name="points")


def clip_positions(layer):
    """The positions of the pair of clipping planes of a layer"""

    return [
        tuple(np.round(plane.position, 3))
        for plane in layer.experimental_clipping_planes
    ]


def plane_event(viewer, position, view_direction):
    """A mouse event on the canvas, looking along a view direction"""

    return SimpleNamespace(
        type="mouse_press",
        pos=np.array([100, 100]),
        modifiers=(),
        position=position,
        view_direction=view_direction,
        dims_displayed=list(viewer.dims.displayed),
        dims_point=list(viewer.dims.point),
        is_dragging=False,
        handled=False,
    )


def canvas_click(layer, viewer, position, view_direction):
    """Press and release the mouse on the canvas, looking along a view direction"""

    event = plane_event(viewer, position, view_direction)
    mouse_press_callbacks(layer, event)
    event.type = "mouse_release"
    mouse_release_callbacks(layer, event)


def test_widget_creation(widget):
    """The plane controls stay hidden until a layer is selected"""

    assert widget.mode == "slice"
    assert widget.orientation_widget.isHidden()
    assert widget.plane_widget.isHidden()
    assert widget.clipping_plane_widget.isHidden()


def test_dock_widget(viewer):
    """The widget is contributed under the name in napari.yaml"""

    _, plugin_widget = viewer.window.add_plugin_dock_widget(
        "napari-plane-sliders", "Plane Sliders"
    )
    assert isinstance(plugin_widget, PlaneSliderWidget)


def test_selecting_a_layer_adds_clipping_planes(widget, viewer, image):
    """Selecting a layer gives it the pair of clipping planes, switched off"""

    viewer.layers.selection.active = image

    assert widget.current_layer is image
    assert len(image.experimental_clipping_planes) == 2
    assert [p.enabled for p in image.experimental_clipping_planes] == [
        False,
        False,
    ]


def test_plane_button_hidden_for_points(widget, viewer, image, points):
    """Points layers have clipping planes but no plane, so plane mode is not offered"""

    viewer.layers.selection.active = image
    assert not widget.plane_btn.isHidden()

    viewer.layers.selection.active = points
    assert widget.plane_btn.isHidden()

    # unless they are linked to a layer that does have a plane
    link_layers([image, points])
    viewer.layers.selection.active = image
    viewer.layers.selection.active = points
    assert not widget.plane_btn.isHidden()


def test_linked_points_layer_drives_the_image_plane(
    widget, viewer, image, points
):
    """The plane slider keeps working while the linked points layer is selected

    That way points can be added on one plane after another without switching
    layers in between.
    """

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.slab_thickness_box.setValue(4)

    viewer.layers.selection.active = points

    # the plane controls stay, the clipping plane range of the points does not
    assert widget.mode == "plane"
    assert not widget.plane_widget.isHidden()
    assert widget.clipping_plane_widget.isHidden()
    assert not widget.slab_widget.isHidden()

    widget.plane_slider.setValue(23)

    assert tuple(image.plane.position) == (23.0, 0.0, 0.0)
    assert clip_positions(points) == [(21.0, 0.0, 0.0), (25.0, 0.0, 0.0)]
    assert all(p.enabled for p in points.experimental_clipping_planes)
    assert not any(p.enabled for p in image.experimental_clipping_planes)

    # and the orientation buttons act on the plane of the image as well
    widget._set_orientation((0, 1, 0))
    assert tuple(image.plane.normal) == (0.0, 1.0, 0.0)


def test_plane_mode(widget, viewer, image):
    """Plane mode sets the depiction and drives the plane position"""

    viewer.layers.selection.active = image
    widget._set_mode("plane")

    assert widget.mode == "plane"
    assert image.depiction == "plane"
    assert widget.plane_slider.isEnabled()
    assert not any(p.enabled for p in image.experimental_clipping_planes)

    widget.plane_slider.setValue(12)
    assert tuple(image.plane.position) == (12.0, 0.0, 0.0)


def test_clipping_plane_mode(widget, viewer, image):
    """Clipping plane mode enables the planes and drives their positions"""

    viewer.layers.selection.active = image
    widget._set_mode("clipping_plane")

    assert widget.mode == "clipping_plane"
    assert image.depiction == "volume"
    assert all(p.enabled for p in image.experimental_clipping_planes)

    widget.clipping_plane_slider.setValue((5, 21))
    assert clip_positions(image) == [(5.0, 0.0, 0.0), (21.0, 0.0, 0.0)]


def test_volume_mode_switches_everything_off(widget, viewer, image):
    """Volume mode returns the layer to a plain volume rendering"""

    viewer.layers.selection.active = image
    widget._set_mode("clipping_plane")
    widget._set_mode("volume")

    assert widget.mode == "volume"
    assert image.depiction == "volume"
    assert not any(p.enabled for p in image.experimental_clipping_planes)


def test_settings_are_remembered_per_layer(widget, viewer, image, points):
    """Switching modes or layers restores the settings stored on each layer"""

    viewer.layers.selection.active = image
    widget._set_mode("plane")
    widget.plane_slider.setValue(7)
    widget._set_mode("clipping_plane")
    widget.clipping_plane_slider.setValue((4, 22))

    # a round trip through volume mode keeps both settings
    widget._set_mode("volume")
    widget._set_mode("plane")
    assert widget.plane_slider.value() == 7
    widget._set_mode("clipping_plane")
    assert tuple(widget.clipping_plane_slider.value()) == (4, 22)

    # and another layer keeps its own
    viewer.layers.selection.active = points
    widget._set_mode("clipping_plane")
    widget.clipping_plane_slider.setValue((9, 17))

    viewer.layers.selection.active = image
    assert tuple(widget.clipping_plane_slider.value()) == (4, 22)
    viewer.layers.selection.active = points
    assert tuple(widget.clipping_plane_slider.value()) == (9, 17)


def test_linked_layers_share_the_clipping_planes(
    widget, viewer, image, points
):
    """Layers the user linked follow the clipping planes of the selected layer"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_mode("clipping_plane")
    widget.clipping_plane_slider.setValue((6, 18))

    assert clip_positions(points) == clip_positions(image)
    assert all(p.enabled for p in points.experimental_clipping_planes)


def test_linked_points_are_clipped_to_a_slab(widget, viewer, image, points):
    """A linked points layer mimics plane mode with a slab around the plane"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.slab_thickness_box.setValue(6)
    widget.plane_slider.setValue(20)

    assert clip_positions(points) == [(17.0, 0.0, 0.0), (23.0, 0.0, 0.0)]
    assert all(p.enabled for p in points.experimental_clipping_planes)
    # the image shows the plane itself, so it must not be clipped
    assert not any(p.enabled for p in image.experimental_clipping_planes)


def test_slab_thickness_only_shown_when_useful(widget, viewer, image, points):
    """The slab thickness applies to linked layers without a plane only"""

    viewer.layers.selection.active = image
    widget._set_mode("plane")
    assert widget.slab_widget.isHidden()

    link_layers([image, points])
    viewer.layers.selection.active = points
    viewer.layers.selection.active = image
    assert not widget.slab_widget.isHidden()


def test_linked_layers_share_the_plane(widget, viewer, image, points):
    """Linked layers that have a plane follow the plane of the selected layer

    napari only links attributes that all linked layers share, so as soon as a
    points layer joins the group the plane is no longer linked by napari itself.
    """

    labels = viewer.add_labels(
        np.zeros((30, 40, 50), dtype=int), name="labels"
    )
    link_layers([image, labels, points])

    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.plane_slider.setValue(14)

    assert labels.depiction == "plane"
    assert tuple(labels.plane.position) == tuple(image.plane.position)
    assert tuple(labels.plane.normal) == tuple(image.plane.normal)


def test_new_points_land_on_the_linked_plane(widget, viewer, image, points):
    """A point added to a linked points layer lands on the plane of the image"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))  # normal along the first axis
    widget._set_mode("plane")
    widget.plane_slider.setValue(20)

    viewer.layers.selection.active = points
    points.mode = "add"

    # a click looking straight at the plane, from outside the volume
    canvas_click(points, viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))

    assert len(points.data) == 21
    added = points.data[-1]
    assert tuple(added) == (20.0, 15.0, 25.0)  # on the plane, under the cursor


def test_points_are_not_snapped_without_a_plane(widget, viewer, image, points):
    """Without a linked layer in plane mode napari places the point as usual"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_mode("volume")

    viewer.layers.selection.active = points
    points.mode = "add"
    canvas_click(points, viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))

    assert len(points.data) == 21
    assert tuple(points.data[-1]) == (
        -10.0,
        15.0,
        25.0,
    )  # the clicked position


def test_points_missing_the_plane_are_not_added(widget, viewer, image, points):
    """A click next to the image adds nothing, rather than a point in empty space"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")

    viewer.layers.selection.active = points
    points.mode = "add"
    canvas_click(points, viewer, (-10.0, 500.0, 500.0), (1.0, 0.0, 0.0))

    assert len(points.data) == 20


def test_point_snapping_follows_the_mode_of_the_layer(
    widget, viewer, image, points
):
    """The snapping callback is only bound while the layer adds points"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    viewer.layers.selection.active = points

    points.mode = "add"
    assert widget._add_point_on_plane in points.mouse_drag_callbacks

    points.mode = "pan_zoom"
    assert widget._add_point_on_plane not in points.mouse_drag_callbacks


def test_moving_the_plane_does_not_reslice_the_points(
    widget, viewer, image, points
):
    """The clipping planes leave the data alone, so the layer must not be sliced again

    Slicing rebuilds the arrays that the status bar thread reads, which makes it fail
    with a shape mismatch when it happens to read them halfway.
    """

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")

    slices, redraws = [], []
    original_set_view_slice = points.set_view_slice
    points.set_view_slice = lambda: (
        slices.append(1),
        original_set_view_slice(),
    )
    points.events.set_data.connect(lambda event: redraws.append(1))

    widget.plane_slider.setValue(18)
    widget.slab_thickness_box.setValue(8)
    widget._set_mode("volume")

    assert redraws  # the visual is told to redraw
    assert not slices  # but the layer is not sliced again


def test_point_mode_does_not_lock_the_linked_image(
    widget, viewer, image, points
):
    """The mode of the points layer must not block panning on the image

    napari links mouse_pan and mouse_zoom between linked layers but not mode, so a
    points layer in select mode would otherwise lock the camera on the image while
    the image keeps its own pan and zoom mode.
    """

    link_layers([image, points])
    viewer.layers.selection.active = points
    points.mode = "select"

    assert not points.mouse_pan  # napari blocks panning while selecting points
    assert image.mouse_pan
    assert image.mouse_zoom


def test_camera_lock_is_lifted_on_the_next_selection(
    widget, viewer, image, points
):
    """A lock left behind on a layer is lifted, also after unlinking the layers"""

    link_layers([image, points])
    viewer.layers.selection.active = points

    # the state napari leaves on the image when it was linked while selecting points
    image.mouse_pan = False
    image.mouse_zoom = False
    unlink_layers([image, points])

    viewer.layers.selection.active = image

    assert image.mouse_pan
    assert image.mouse_zoom
    assert viewer.camera.mouse_pan


def test_empty_linked_points_layer(widget, viewer, image):
    """An empty points layer has no extent of its own, which must not break the sliders"""

    empty = viewer.add_points(name="empty")
    link_layers([image, empty])

    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.plane_slider.setValue(20)

    # selecting the empty layer used to raise on the NaN extent
    viewer.layers.selection.active = empty

    # the slider covers the image, and the slab around its plane survives
    assert widget.plane_slider.maximum() == 29
    assert widget.plane_slider.value() == 20
    assert clip_positions(empty) == [(17.5, 0.0, 0.0), (22.5, 0.0, 0.0)]


def test_lone_empty_points_layer(widget, viewer):
    """An empty points layer on its own has neither an extent nor three dimensions"""

    empty = viewer.add_points(name="empty")
    viewer.layers.selection.active = empty
    widget._set_mode("clipping_plane")

    assert widget.mode == "clipping_plane"
    assert len(empty.experimental_clipping_planes) == 2


def test_sliders_span_the_whole_linked_group(widget, viewer, image):
    """A points layer that covers a small part of the image still spans the image"""

    few = viewer.add_points(np.array([[10.0, 20.0, 25.0]]), name="few")
    link_layers([image, few])

    viewer.layers.selection.active = few
    widget._set_mode("clipping_plane")

    assert widget.clipping_plane_slider.maximum() == 29


def test_click_position_snaps_to_the_plane(widget, viewer, image, points):
    """The reported cursor position is on the plane, for the layer and its linked layers"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.plane_slider.setValue(20)

    event = plane_event(viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    widget._snap_cursor_to_plane(viewer, event)
    assert tuple(viewer.cursor.position) == (20.0, 15.0, 25.0)

    # a linked layer without a plane of its own snaps to the same plane
    viewer.cursor.position = (0.0, 0.0, 0.0)
    viewer.layers.selection.active = points
    widget._snap_cursor_to_plane(viewer, event)
    assert tuple(viewer.cursor.position) == (20.0, 15.0, 25.0)


def test_hover_position_snaps_to_the_plane(widget, viewer, image):
    """Hovering reports the position on the plane, so shortcuts can act on it

    napari-orthogonal-views binds a key to a function that reads
    `viewer.cursor.position` under the mouse, which only ever sees the hover
    position, not a click.
    """

    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.plane_slider.setValue(20)

    event = plane_event(viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    event.type = "mouse_move"
    mouse_move_callbacks(viewer, event)

    assert tuple(viewer.cursor.position) == (20.0, 15.0, 25.0)

    # what napari-orthogonal-views does with it
    step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                    viewer.cursor.position, viewer.dims.range, strict=False
                )
            ]
        ).astype(int)
    )
    assert step == (20, 15, 25)  # the slice the plane is on


def test_snapping_can_be_switched_off(widget, viewer, image, points):
    """The snap to plane checkbox turns off both the position and the point snapping"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_orientation((1, 0, 0))
    widget._set_mode("plane")
    widget.plane_slider.setValue(20)
    assert widget.snap_checkbox.isChecked()  # on by default

    viewer.layers.selection.active = points
    points.mode = "add"
    widget.snap_checkbox.setChecked(False)

    # the position is reported as napari computes it
    viewer.cursor.position = (1.0, 2.0, 3.0)
    event = plane_event(viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    event.type = "mouse_move"
    mouse_move_callbacks(viewer, event)
    assert tuple(viewer.cursor.position) == (1.0, 2.0, 3.0)

    # and points are added where napari puts them
    assert widget._add_point_on_plane not in points.mouse_drag_callbacks
    canvas_click(points, viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    assert tuple(points.data[-1]) == (-10.0, 15.0, 25.0)

    # switching it back on restores both
    widget.snap_checkbox.setChecked(True)
    assert widget._add_point_on_plane in points.mouse_drag_callbacks
    canvas_click(points, viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    assert tuple(points.data[-1]) == (20.0, 15.0, 25.0)


def test_click_position_kept_without_a_plane(widget, viewer, image):
    """Clicks are left alone when no plane is shown"""

    viewer.layers.selection.active = image
    widget._set_mode("volume")

    viewer.cursor.position = (1.0, 2.0, 3.0)
    widget._snap_cursor_to_plane(
        viewer, plane_event(viewer, (-10.0, 15.0, 25.0), (1.0, 0.0, 0.0))
    )
    assert tuple(viewer.cursor.position) == (1.0, 2.0, 3.0)


def test_controls_hidden_in_2d(widget, viewer, image):
    """The plane controls only make sense in 3D"""

    viewer.layers.selection.active = image
    widget._set_mode("plane")

    viewer.dims.ndisplay = 2

    assert widget.mode == "slice"
    assert widget.orientation_widget.isHidden()
    assert widget.plane_widget.isHidden()
    assert widget.clipping_plane_widget.isHidden()


def test_mode_buttons_switch_to_3d(widget, viewer, image):
    """Clicking a mode button in 2D switches the viewer to 3D and applies the mode"""

    viewer.dims.ndisplay = 2
    viewer.layers.selection.active = image
    assert widget.isEnabled()
    assert not any(b.isChecked() for b in widget._mode_buttons.values())

    widget.plane_btn.click()

    assert viewer.dims.ndisplay == 3
    assert widget.mode == "plane"
    assert image.depiction == "plane"
    assert widget.plane_btn.isChecked()
    assert not widget.volume_btn.isChecked()

    viewer.dims.ndisplay = 2
    widget.clipping_plane_btn.click()

    assert viewer.dims.ndisplay == 3
    assert widget.mode == "clipping_plane"
    assert image.depiction == "volume"
    assert all(p.enabled for p in image.experimental_clipping_planes)


def test_widget_disabled_without_a_layer(widget, viewer, image):
    """The controls only become active once a supported layer is selected"""

    viewer.layers.selection.clear()
    widget_without_layer = type(widget)(viewer)
    assert not widget_without_layer.isEnabled()

    viewer.layers.selection.active = image
    assert widget_without_layer.isEnabled()


def test_changing_orientation_moves_the_plane(widget, viewer):
    """The plane follows a new normal even if the slider value stays the same"""

    image = viewer.add_image(np.random.random((30, 30, 30)))
    viewer.layers.selection.active = image
    widget._set_mode("plane")
    widget._set_orientation((1, 0, 0))
    assert tuple(image.plane.position) == (14.0, 0.0, 0.0)

    widget._set_orientation((0, 1, 0))
    assert tuple(image.plane.position) == (0.0, 14.0, 0.0)


def test_depiction_changed_in_layer_controls(widget, viewer, image, points):
    """Changing the depiction outside the widget switches the mode along"""

    link_layers([image, points])
    viewer.layers.selection.active = points
    widget._set_mode("plane")

    image.depiction = "volume"
    assert widget.mode == "volume"

    image.depiction = "plane"
    assert widget.mode == "plane"


def test_no_snapping_in_2d(widget, viewer, image, points):
    """In 2D there is no plane to snap to, points are added the napari way"""

    link_layers([image, points])
    viewer.layers.selection.active = points
    widget._set_mode("plane")
    assert widget._plane_layer_for(points) is image

    viewer.dims.ndisplay = 2
    assert widget._plane_layer_for(points) is None
