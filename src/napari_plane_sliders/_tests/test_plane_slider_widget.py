import numpy as np
import pytest
from napari.experimental import link_layers

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


def test_widget_creation(widget):
    """The plane controls stay hidden until a layer is selected"""

    assert widget.mode == "slice"
    assert widget.plane_labels.isHidden()
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


def test_plane_mode(widget, viewer, image):
    """Plane mode sets the depiction and drives the plane position"""

    viewer.layers.selection.active = image
    widget._set_plane_mode()

    assert widget.mode == "plane"
    assert image.depiction == "plane"
    assert widget.plane_slider.isEnabled()
    assert not any(p.enabled for p in image.experimental_clipping_planes)

    widget.plane_slider.setValue(12)
    assert tuple(image.plane.position) == (12.0, 0.0, 0.0)


def test_clipping_plane_mode(widget, viewer, image):
    """Clipping plane mode enables the planes and drives their positions"""

    viewer.layers.selection.active = image
    widget._set_clipping_plane_mode()

    assert widget.mode == "clipping_plane"
    assert image.depiction == "volume"
    assert all(p.enabled for p in image.experimental_clipping_planes)

    widget.clipping_plane_slider.setValue((5, 21))
    assert clip_positions(image) == [(5.0, 0.0, 0.0), (21.0, 0.0, 0.0)]


def test_volume_mode_switches_everything_off(widget, viewer, image):
    """Volume mode returns the layer to a plain volume rendering"""

    viewer.layers.selection.active = image
    widget._set_clipping_plane_mode()
    widget._set_volume_mode()

    assert widget.mode == "volume"
    assert image.depiction == "volume"
    assert not any(p.enabled for p in image.experimental_clipping_planes)


def test_settings_are_remembered_per_layer(widget, viewer, image, points):
    """Switching modes or layers restores the settings stored on each layer"""

    viewer.layers.selection.active = image
    widget._set_plane_mode()
    widget.plane_slider.setValue(7)
    widget._set_clipping_plane_mode()
    widget.clipping_plane_slider.setValue((4, 22))

    # a round trip through volume mode keeps both settings
    widget._set_volume_mode()
    widget._set_plane_mode()
    assert widget.plane_slider.value() == 7
    widget._set_clipping_plane_mode()
    assert tuple(widget.clipping_plane_slider.value()) == (4, 22)

    # and another layer keeps its own
    viewer.layers.selection.active = points
    widget._set_clipping_plane_mode()
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
    widget._set_clipping_plane_mode()
    widget.clipping_plane_slider.setValue((6, 18))

    assert clip_positions(points) == clip_positions(image)
    assert all(p.enabled for p in points.experimental_clipping_planes)


def test_linked_points_are_clipped_to_a_slab(widget, viewer, image, points):
    """A linked points layer mimics plane mode with a slab around the plane"""

    link_layers([image, points])
    viewer.layers.selection.active = image
    widget._set_z_orientation()
    widget._set_plane_mode()
    widget.slab_thickness_box.setValue(6)
    widget.plane_slider.setValue(20)

    assert clip_positions(points) == [(17.0, 0.0, 0.0), (23.0, 0.0, 0.0)]
    assert all(p.enabled for p in points.experimental_clipping_planes)
    # the image shows the plane itself, so it must not be clipped
    assert not any(p.enabled for p in image.experimental_clipping_planes)


def test_slab_thickness_only_shown_when_useful(widget, viewer, image, points):
    """The slab thickness applies to linked layers without a plane only"""

    viewer.layers.selection.active = image
    widget._set_plane_mode()
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
    widget._set_z_orientation()
    widget._set_plane_mode()
    widget.plane_slider.setValue(14)

    assert labels.depiction == "plane"
    assert tuple(labels.plane.position) == tuple(image.plane.position)
    assert tuple(labels.plane.normal) == tuple(image.plane.normal)


def test_controls_hidden_in_2d(widget, viewer, image):
    """The plane controls only make sense in 3D"""

    viewer.layers.selection.active = image
    widget._set_plane_mode()

    viewer.dims.ndisplay = 2

    assert widget.mode == "slice"
    assert widget.plane_labels.isHidden()
    assert widget.plane_widget.isHidden()
    assert widget.clipping_plane_widget.isHidden()
