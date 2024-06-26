import math

WRIST_ROTATE_CONSTANT = 0.395

def normalize_angles(degree_values):
    # print(type(degree_values))
    degree_ranges = [
        (-175, 175),
        (-265, 85),
        (-160, 160),
        (-265, 85),
        (-175, 175),
        (-175, 175)
    ]
    
    radian_ranges = [
        (-3.05433, 3.05433),
        (-4.62512, 1.48353),
        (-2.82743, 2.82743),
        (-4.62512, 1.48353),
        (-3.14158, 3.14158),
        (-3.14158, 3.14158)
    ]
    
    normalized_values = []

    joint_list = degree_values[1]

    for i in range(6):
        degree_min, degree_max = degree_ranges[i]
        radian_min, radian_max = radian_ranges[i]
        
        normalized_value = radian_min + (radian_max - radian_min) * (joint_list[i] - degree_min) / (degree_max - degree_min)
        normalized_values.append(normalized_value)
    
    return normalized_values