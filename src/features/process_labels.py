import pandas as pd


def find_label(movement):
    if movement == "Walking":
        return "walking"
    if movement == "Moving arm faster towards radar, slower away":
        return "pushing"
    if movement == "Sitting and standing":
        return "sitting"
    if movement == "Moving arm slower towards radar, faster away":
        return "pulling"
    if movement == "Circling arm forwards":
        return "circling"
    if movement == "Clapping":
        return "clapping"
    if movement == "Bending to pick up and back up":
        return "bending"


def identify_angle(angle):
    return angle.split()[0]


def is_on_place(angle):
    if len(angle.split()) > 2:
        return True
    return False


def assign_user_label(name):
    if name == "Aleksandar":
        return "A"
    if name == "Francesco":
        return "B"
    if name == "Nadezhda":
        return "C"
    if name == "Leila":
        return "D"
    if name == "Hadi":
        return "E"
    if name == "Ivelina":
        return "F"


def process_labels(df_labels):
    df_labels["label"] = df_labels.movement.apply(find_label)
    df_labels["user_label"] = df_labels.person.apply(assign_user_label)
    df_labels["aspect_angle"] = df_labels.angle.apply(identify_angle)
    df_labels["on_place"] = df_labels.angle.apply(is_on_place)
    return df_labels
