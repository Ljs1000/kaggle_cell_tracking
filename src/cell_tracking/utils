import napari

def view_with_tracks(image_arr, tracks_arr, image_name="cells", tracks_name="ground_truth_tracks"):
    viewer = napari.Viewer()
    viewer.add_image(image_arr, name=image_name)
    viewer.add_tracks(tracks_arr, name=tracks_name)
    return viewer