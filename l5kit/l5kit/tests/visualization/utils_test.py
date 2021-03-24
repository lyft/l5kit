import numpy as np

from l5kit.visualization import draw_path_prior_layer, draw_reference_trajectory, draw_trajectory


def test_draw_trajectory() -> None:
    on_image = np.zeros((224, 244, 3), dtype=np.uint8)
    positions = np.asarray([(0, 0), (0, 10), (0, 20)])  # XY notation, pixel positions
    draw_trajectory(on_image, positions, (255, 255, 255))

    assert np.all(on_image[0, 0] == (255, 255, 255))
    assert np.all(on_image[10, 0] == (255, 255, 255))
    assert np.all(on_image[20, 0] == (255, 255, 255))

    assert np.all(on_image[0, 20] == (0, 0, 0))
    assert np.all(on_image[0, 10] == (0, 0, 0))

    # test also with arrowed lines
    on_image = np.zeros((224, 244, 3), dtype=np.uint8)
    yaws = np.asarray([[0.1], [-0.1], [0.0]])
    draw_trajectory(on_image, positions, (255, 255, 255), yaws=yaws)

    assert np.all(on_image[0, 0] == (255, 255, 255))
    assert np.all(on_image[10, 0] == (255, 255, 255))
    assert np.all(on_image[20, 0] == (255, 255, 255))


def test_draw_reference_trajectory() -> None:
    on_image = np.zeros((224, 244, 3), dtype=np.uint8)
    positions = np.asarray([(0, 0), (1, 1), (2, 2)])  # XY notation, meter absolute
    world_to_pixel = np.asarray(((10, 0, 112), (0, 10, 112), (0, 0, 1)))  # 1m->10px
    draw_reference_trajectory(on_image, world_to_pixel, positions)

    assert np.all(on_image[112, 112] == (255, 255, 0))
    assert np.all(on_image[112 + 10, 112 + 10] == (255, 255, 0))
    assert np.all(on_image[112 + 20, 112 + 20] == (255, 255, 0))


def test_draw_prior() -> None:
    positions = np.asarray([(0, 0), (1, 1), (2, 2)])  # XY notation, meter absolute
    world_to_pixel = np.asarray(((10, 0, 112), (0, 10, 112), (0, 0, 1)))  # 1m->10px
    layer = draw_path_prior_layer((224, 224), world_to_pixel, positions)

    assert np.all(layer[112, 112] == 1.0)
    assert np.all(layer[112 + 10, 112 + 10] == 1.0)
    assert np.all(layer[112 + 20, 112 + 20] == 1.0)

    assert np.all(layer[0, 0] == 0.0)
