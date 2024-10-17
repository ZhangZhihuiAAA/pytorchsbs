import numpy as np


def gen_img(start, target, fill=1, img_size=10):
    # Generate empty image
    img = np.zeros((img_size, img_size), dtype=float)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    # Draw non-diagonal line if target = 0
    if target == 0:
        if start_row is None:
            # Draw vertical line if target = 0 and start <= 0
            img[:, start_col] = fill
        else:
            # Draw horizontal line if target = 0 and start > 0
            img[start_row, :] = fill
    # Draw diagonal line if target != 0
    else:
        if target == 1:
            if start_row is not None:
                # Draw up from left border to top border if target = 1 and start > 0
                up = (range(start_row, -1, -1),
                      range(0, start_row + 1))
            else:
                # Draw up from bottom border to right border if target = 1 and start <= 0
                up = (range(img_size - 1, start_col - 1, -1),
                      range(start_col, img_size))
            img[up] = fill
        else:
            if start_row is not None:
                # Draw down from left border to bottom border if target not in [0, 1] and start > 0
                down = (range(start_row, img_size, 1), 
                        range(0, img_size - start_row))
            else:
                # Draw down from top border to right border if target not in [0, 1] adn start <= 0
                down = (range(0, img_size - start_col), 
                        range(start_col, img_size))
            img[down] = fill

    return 255 * img.reshape(1, img_size, img_size)


def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
    rng = np.random.default_rng(seed)

    starts = rng.integers(-(img_size - 1), img_size, size=(n_images,))
    targets = rng.integers(0, 3, size=(n_images,))

    images = np.array([gen_img(s, t, img_size=img_size) for s, t in zip(starts, targets)], dtype=np.uint8)

    if binary:
        targets = (targets > 0).astype(int)

    return images, targets