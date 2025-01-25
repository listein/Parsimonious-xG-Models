import numpy as np
import math
from shapely.geometry import Point, Polygon

#convert yards to meter
def yard_to_meter(yard):
    return yard/1.09361

#calculate Euclidean Distance
def get_distance(x, y):
    return math.sqrt((120-x)**2 + (40-y)**2) #distance to center of goal at (120,40)

#calculate angle with formula cos(phi) = (a*b)/(|a|*|b|)
def get_angle(x, y):
    a, b = [[120 - x, 36 - y], [120 - x, 44 - y]]
    #shooting angle
    dot_prod = np.dot(a, b)
    magn_a = np.linalg.norm(a)
    magn_b = np.linalg.norm(b)

    theta = dot_prod / (magn_a * magn_b)

    rad_theta = np.arccos(np.clip(theta, -1.0, 1.0))

    #angle between shooting position to middle of goal and perpendicular line (displacement of player)
    c, d = [[120-x, 40-y], [1, 0]]

    dot_prod = np.dot(c, d)
    magn_c = np.linalg.norm(c)
    magn_d = np.linalg.norm(d)

    phi = dot_prod / (magn_c * magn_d)

    rad_phi = np.arccos(np.clip(phi, -1.0, 1.0))

    return np.degrees(rad_theta), np.degrees(rad_phi)

def get_interval(ball_loc, player_loc):
    # Convert to numpy arrays
    ball_loc = np.array([ball_loc.x, ball_loc.y])
    player_loc = np.array([player_loc.x, player_loc.y])

    # First, get the distance vector between ball and player
    d_vec = player_loc - ball_loc
    distance = np.linalg.norm(d_vec)

    #player to close, simplify by assuming he blocks whole goal
    if distance <= 0.5:
        return [36, 44]

    theta = np.arcsin(0.5/distance) #angle between tangential line and d_vec

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x = d_vec[0] #x component of vector between ball and player
    y = d_vec[1] #y component of vector between ball and player

    y_u = ball_loc[1] + (120 - ball_loc[0]) * (sin_t*x + cos_t*y) / (cos_t*x - sin_t*y)
    y_l = ball_loc[1] + (120 - ball_loc[0]) * (-sin_t*x + cos_t*y) / (cos_t*x + sin_t*y)

    return [y_l, y_u]

def merge_interval(intervals):
    n = len(intervals)

    intervals.sort()
    res = []

    for i in range(n):
        #no value lower than 36 or larger than 44 is taken (outside of the goal)
        start = max(36, intervals[i][0])
        end = min(44, intervals[i][1])

        if start >= 44 or end <= 36:
            continue

        # Skipping already merged intervals
        if res and res[-1][1] >= end:
            continue

        # Find the end of the merged range
        for j in range(i + 1, n):
            if intervals[j][0] <= end:
                end = min(44, max(end, intervals[j][1]))
        res.append([start, end])
    return res

def get_visible_angle(x, y, freeze_frame, angle):
    #store each interval blocked by a player
    intervals = []

    #use geometry to determine if opponents are inside the triangle formed by goal and ball
    ball_loc = Point(x, y)
    triangle = Polygon([(120, 36), (120, 44), ball_loc])

    for player in freeze_frame:
        player_loc = Point(player['location'][0], player['location'][1])
        loc_buf = player_loc.buffer(1, 1) #diameter of player is 1yd

        #True when player is in front of the ball location and inside the triangle
        if loc_buf.intersects(triangle) and ball_loc.x < player_loc.x:
            intervals.append(get_interval(ball_loc, player_loc))

    #merge all the intervals
    interval = merge_interval(intervals)

    #same approach used get_angle() but shortened to one line
    total_angle = [np.degrees(np.arccos(np.clip(np.dot([120 - x, start - y], [120 - x, end - y]) / (np.linalg.norm([120 - x, start - y]) * np.linalg.norm([120 - x, end - y])), -1.0, 1.0))) for start, end in interval]

    #total angle is the blocked part, but we want the visible part from the shooting angle
    vis_angle = angle - sum(total_angle)
    return vis_angle
