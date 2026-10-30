from math import sqrt


# ----------------------------------------------------------------------------
# Freeman 8-direction chain code
#
#        3  2  1
#         \ | /
#      4 -- P -- 0        (image coordinates: x to the right, y DOWN)
#         / | \
#        5  6  7
#
# 0 = east, then counter-clockwise as seen on screen. "Up" on screen is y - 1.
# ----------------------------------------------------------------------------

# code -> (dx, dy)
DIRECTIONS = {
    0: (1, 0),
    1: (1, -1),
    2: (0, -1),
    3: (-1, -1),
    4: (-1, 0),
    5: (-1, 1),
    6: (0, 1),
    7: (1, 1),
}

# (dx, dy) -> code
DELTA_TO_CODE = {delta: code for code, delta in DIRECTIONS.items()} #gives them in reversed order
#  { 
# (1,0):0,
# (1,-1): 1
# ...
#  }) #i have a coordinate offset and want to get the code representing it

#We cannot draw a perfectly continuous mathematical line on a digital image because pixels are discrete.
#We need to choose which pixels should be turned on to approximate the line.

def bresenham_line(p0, p1): #connects two points only with a pixel line
    """
    All pixels on the straight line from p0 to p1, both included.
    Consecutive pixels always touch (8-connected).
    """
    x0, y0 = int(p0[0]), int(p0[1]) #Converts the endpoints to integer pixel coordinates.
    x1, y1 = int(p1[0]), int(p1[1])

    dx = abs(x1 - x0) #the absolute distances to travel in each axis
    dy = abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1 #right or left along the x axis
    step_y = 1 if y0 < y1 else -1 #up or down along the y axis

    error = dx - dy 

    #tracks how far the current pixel is from the ideal line, in a scaled-up integer form. 
    #The ideal line satisfies dy * (x - x_start) = dx * (y - y_start), (normal line equation)
    # and 'error' accumulates the difference between those two quantities.
    # its a measure of how much horizontal movement is still needed compared with vertical movement.

    
    pixels = [] #This list stores every pixel that belongs to the line.

    while True:
        pixels.append((x0, y0))

        if x0 == x1 and y0 == y1: #stop once we've reached the endpoint
            break

        doubled_error = 2 * error #to avoid floating-point calculations

        if doubled_error > -dy:   # moving horizontally
            error -= dy
            x0 += step_x

        if doubled_error < dx:    # moving vertically
            error += dx
            y0 += step_y

            '''
            the if conditions are independent, meaning both conditions can be true, so we move diagonally (one pixel in x and one in y).
            '''

    return pixels


def contour_to_pixel_path(contour):
    """
    Turn the snake's sparse points into a closed path where every pixel
    touches the next one. Returns a list of (x, y) with no repeated pixel.
    """
    points = [(int(x), int(y)) for x, y in contour]

    # 1) remove consecutive duplicates (the snake can pile points on one pixel)
    cleaned = [points[0]]
    for p in points[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)

    # the contour is closed, so the last point may equal the first
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(cleaned) < 2:
        raise ValueError("Contour needs at least 2 distinct points.")

    # 2) connect point i to point i+1, and the last one back to the first
    path = []
    n = len(cleaned)
    for i in range(n):
        segment = bresenham_line(cleaned[i], cleaned[(i + 1) % n])
        # drop the last pixel: it is the first pixel of the next segment
        path.extend(segment[:-1])

    return path


def pixel_path_to_chain_code(path):
    """
    Closed pixel path -> (start_pixel, list of codes).
    The last code is the step from the last pixel back to the start,
    so len(codes) == len(path) and the code describes a closed curve.
    """
    codes = []
    n = len(path)

    for i in range(n):
        x, y = path[i]
        next_x, next_y = path[(i + 1) % n]
        delta = (next_x - x, next_y - y)

        if delta not in DELTA_TO_CODE:
            raise ValueError(f"Pixels {path[i]} and {path[(i + 1) % n]} are not 8-neighbours.")

        codes.append(DELTA_TO_CODE[delta])

    return path[0], codes


def contour_to_chain_code(contour):
    """Snake points -> (start_pixel, codes), For the API call"""
    path = contour_to_pixel_path(contour)
    return pixel_path_to_chain_code(path)


def compute_perimeter(chain_code):
    """Calculate the contour perimeter in pixels from its Freeman chain code."""
    if not chain_code:
        return 0.0
    if any(code not in range(8) for code in chain_code):
        raise ValueError("Chain code must contain values from 0 to 7.")

    return sum(sqrt(2) if code % 2 else 1 for code in chain_code)


def compute_area(chain_code, start_point=(0, 0)):
    """Calculate enclosed area in square pixels using the shoelace formula."""
    if not chain_code:
        return 0.0
    if any(code not in DIRECTIONS for code in chain_code):
        raise ValueError("Chain code must contain values from 0 to 7.")

    x, y = start_point
    start_x, start_y = x, y
    area_sum = 0

    for code in chain_code:
        dx, dy = DIRECTIONS[code]
        next_x, next_y = x + dx, y + dy
        area_sum += x * next_y - next_x * y
        x, y = next_x, next_y

    if (x, y) != (start_x, start_y):
        raise ValueError("Chain code does not describe a closed contour.")

    return abs(area_sum) / 2