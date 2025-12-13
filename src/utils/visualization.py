import matplotlib.pyplot as plt

def show_image_with_bbox(img_tensor, bbox, img_size=640, ax=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    cx, cy, w, h = bbox
    cx *= img_size
    cy *= img_size
    w  *= img_size
    h  *= img_size

    x1 = cx - w / 2
    y1 = cy - h / 2

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    ax.add_patch(plt.Rectangle(
        (x1, y1), w, h,
        edgecolor="green", facecolor="none", linewidth=2
    ))
    ax.axis("off")
