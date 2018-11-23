import os


# make a directory path for the spectrograms to go in
def make_directory(interim_path, window_size, user_label, angle_label, action_label):
    #  interim/window_size/user_label/angle_label/action_label
    window_directory = interim_path + str(window_size)
    if not os.path.exists(window_directory):
        os.makedirs(window_directory)

    user_directory = window_directory + "/" + user_label
    if not os.path.exists(user_directory):
        os.makedirs(user_directory)

    angle_directory = user_directory + "/" + angle_label
    if not os.path.exists(angle_directory):
        os.makedirs(angle_directory)

    action_directory = angle_directory + "/" + action_label
    if not os.path.exists(action_directory):
        os.makedirs(action_directory)

    return action_directory
