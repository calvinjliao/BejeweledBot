import os
import random
import _thread
from typing import Tuple, List
from time import sleep

import cv2
import keyboard
import numpy as np
import pyautogui
from python_imagesearch.imagesearch import imagesearch_numLoop, imagesearch_region_loop, \
    region_grabber

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Directories
IMAGE_DIRECTORY = './images'
GEM_DIRECTORY = './images/gems'

# Game window dimensions [x,y]
GAME_WINDOW_DIM = (1089, 817)
# Pixel offset for where the grid starts
GAME_GRID_OFFSET = (356, 51)
# Number of tiles in the grid [x,y]
GAME_GRID_TILES = (8, 8)
# Size of each tile in the grid in pixels [x,y]
GAME_TILE_DIM = (87, 87)
# Neural Net
Gem_Neural_Net = None
# Game Region
Game_Region = None

# Gem to number references
GEM_MAP = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'hypercube']
TILE_MAP = {'red': 0, 'orange': 1, 'yellow': 2, 'green': 3, 'blue': 4, 'purple': 5, 'white': 6, 'hypercube': 7}


# Shortcut for getting image path
def get_image(directory: str, image: str):
    return np.os.path.join(directory, image)


# Searches for game window
def find_game():
    return imagesearch_numLoop(get_image(IMAGE_DIRECTORY, 'menu.png'), maxSamples=50, timesample=0.1, precision=0.8)


# Automation function to navigate to Zen Mode when the game window is found
def start_game(game_region: List[int]):

    x1 = game_region[0]
    y1 = game_region[1]
    x2 = game_region[0] + GAME_WINDOW_DIM[0]
    y2 = game_region[1] + GAME_WINDOW_DIM[1]
    start_pos = imagesearch_region_loop(get_image(IMAGE_DIRECTORY, 'play.png'), 0.1, x1, y1, x2, y2, precision=0.9)
    if start_pos[0] == -1:
        raise RuntimeError('Play button not found')
    pyautogui.click(game_region[0] + start_pos[0] + 60, game_region[1] + start_pos[1] + 18)
    sleep(1)
    zen_pos = imagesearch_region_loop(get_image(IMAGE_DIRECTORY, 'zen.png'), 0.1, x1, y1, x2, y2, precision=0.9)
    if zen_pos[0] == -1:
        raise RuntimeError('Zen button not found')
    pyautogui.click(game_region[0] + zen_pos[0] + 60, game_region[1] + zen_pos[1] + 18)


# Initializes an empty playing grid
def initialize() -> np.core.multiarray:
    return np.empty(GAME_GRID_TILES[0], dtype=np.empty(GAME_GRID_TILES[1], dtype=int))


# Circuit breaking class
# I have a feeling there's better ways to do this, but I need to capture input from any window
# so this will have to do
class Breaker:
    b = None

    def __init__(self, breaker):
        self.b = breaker

    def __call__(self, unusued):
        self.b.append(True)

    def get_b(self):
        return self.b


# Circuit Breaker Function
def breaker_thread(b_list):
    print('Breaker thread started')
    x = Breaker(b_list)
    keyboard.on_press(x)


# Main logic. Searches the grid,
def play_loop():
    breaker = []
    _thread.start_new_thread(breaker_thread, (breaker,))
    while not breaker:
        print('new iter')
        state = check_stability()
        grid = identify_tiles(state)
        xpos, ypos, dir = find_match(grid)
        execute_swap(xpos, ypos, dir, grid)
        sleep(0.5)
        # Need to check progress bar somehow?


# Periodically scans the grid, returning the image when the grid settles
def check_stability():
    match_percent = 0
    curr_state = np.array(region_grabber(Game_Region))
    while match_percent < 0.97:
        sleep(0.4)
        new_state = np.array(region_grabber(Game_Region))
        match_percent = cv2.matchTemplate(curr_state, new_state, cv2.TM_CCOEFF_NORMED)
        curr_state = new_state
    return cv2.cvtColor(curr_state, cv2.COLOR_RGB2BGR)


# Divides the grid into individual tiles and uses the neural net to identify them
# Returns the identified tiles, converted to integer format
# See GEM_MAP for conversion
def identify_tiles(state):
    if Gem_Neural_Net is None:
        raise RuntimeError('Neural net not loaded')
    grid_images = []
    for y in range(GAME_GRID_TILES[1]):
        y_start = GAME_TILE_DIM[1] * y
        y_end = y_start + GAME_TILE_DIM[1]
        for x in range(GAME_GRID_TILES[0]):
            x_start = GAME_TILE_DIM[0] * x
            x_end = x_start + GAME_TILE_DIM[0]
            tile = state[x_start:x_end, y_start:y_end]
            grid_images.append(tile)
    grid_images = np.array(grid_images).reshape(-1, GAME_TILE_DIM[0], GAME_TILE_DIM[1], 3)
    grid = Gem_Neural_Net.predict_classes(grid_images)
    return np.array(grid).reshape(GAME_GRID_TILES[0], GAME_GRID_TILES[1])


# Finds the first match starting from the top left
# Returns in format x, y, 0/1 - 0 for right, 1 for down
# Returns None if no matches are found (i.e. only a hypercube match remains)
def find_match(grid) -> Tuple[int, int, str]:
    for y in range(0, GAME_GRID_TILES[1], 1):
        for x in range(0, GAME_GRID_TILES[0], 1):
            # Check for matches when swapped right
            curr_gem = grid[x][y]
            # Check for matches when swapped down
            if y < GAME_GRID_TILES[1] - 1:
                swap_gem = grid[x][y + 1]
                c_match_d = (x > 1 and grid[x - 2][y + 1] == curr_gem,
                             x > 0 and grid[x - 1][y + 1] == curr_gem,
                             x < GAME_GRID_TILES[0] - 1 and grid[x + 1][y + 1] == curr_gem,
                             x < GAME_GRID_TILES[0] - 2 and grid[x + 2][y + 1] == curr_gem,
                             y < GAME_GRID_TILES[1] - 3 and grid[x][y + 2] == curr_gem,
                             y < GAME_GRID_TILES[1] - 3 and grid[x][y + 3] == curr_gem)
                if (c_match_d[1] and (c_match_d[0] or c_match_d[2])) \
                        or (c_match_d[2] and c_match_d[3]) \
                        or (c_match_d[4] and c_match_d[5]):
                    return x, y, 'down'
                s_match_d = (x > 1 and grid[x - 2][y] == swap_gem,
                             x > 0 and grid[x - 1][y] == swap_gem,
                             x < GAME_GRID_TILES[1] - 1 and grid[x + 1][y] == swap_gem,
                             x < GAME_GRID_TILES[1] - 2 and grid[x + 2][y] == swap_gem,
                             y > 1 and grid[x][y - 2] == swap_gem,
                             y > 1 and grid[x][y - 1] == swap_gem)
                if (s_match_d[1] and (s_match_d[0] or s_match_d[2])) \
                        or (s_match_d[2] and s_match_d[3]) \
                        or (s_match_d[4] and s_match_d[5]):
                    return x, y, 'down'
            if x < GAME_GRID_TILES[0] - 1:
                swap_gem = grid[x + 1][y]
                c_match_r = (y > 1 and grid[x + 1][y - 2] == curr_gem,
                             y > 0 and grid[x + 1][y - 1] == curr_gem,
                             y < GAME_GRID_TILES[1] - 1 and grid[x + 1][y + 1] == curr_gem,
                             y < GAME_GRID_TILES[1] - 2 and grid[x + 1][y + 2] == curr_gem,
                             x < GAME_GRID_TILES[0] - 3 and grid[x + 2][y] == curr_gem,
                             x < GAME_GRID_TILES[0] - 3 and grid[x + 3][y] == curr_gem)
                if (c_match_r[1] and (c_match_r[0] or c_match_r[2])) \
                        or (c_match_r[2] and c_match_r[3]) \
                        or (c_match_r[4] and c_match_r[5]):
                    return x, y, 'right'
                s_match_r = (y > 1 and grid[x][y - 2] == swap_gem,
                             y > 0 and grid[x][y - 1] == swap_gem,
                             y < GAME_GRID_TILES[1] - 1 and grid[x][y + 1] == swap_gem,
                             y < GAME_GRID_TILES[1] - 2 and grid[x][y + 2] == swap_gem,
                             x > 1 and grid[x - 2][y] == swap_gem,
                             x > 1 and grid[x - 1][y] == swap_gem)
                if (s_match_r[1] and (s_match_r[0] or s_match_r[2])) \
                        or (s_match_r[2] and s_match_r[3]) \
                        or (s_match_r[4] and s_match_r[5]):
                    return x, y, 'right'
    # No matches besides hypercube
    hypercubes = np.where(grid == 7)
    if hypercubes[0]:
        return hypercubes[0][0], hypercubes[1][0], 'hypercube'
    return None, None, None


# Uses pyautogui to execute a match
# Hypercubes will get swapped to maximize destroying maximum number of gems
def execute_swap(x: int, y: int, dir: str, grid):
    xpos = int(Game_Region[0] + (x + 0.5) * GAME_TILE_DIM[0])
    ypos = int(Game_Region[1] + (y + 0.5) * GAME_TILE_DIM[1])
    pyautogui.click(xpos, ypos)
    if dir == 'right':
        pyautogui.click(xpos + GAME_TILE_DIM[0], ypos, duration=0.2)
        pyautogui.moveTo(Game_Region[0] - 100, Game_Region[1])
    elif dir == 'down':
        pyautogui.click(xpos, ypos + GAME_TILE_DIM[1], duration=0.2)
        pyautogui.moveTo(Game_Region[0] - 100, Game_Region[1])
    elif dir == 'hypercube':
        counts = {}
        if x > 0:
            counts[(grid == grid[x - 1][y]).sum()] = 'left'
        if y > 0:
            counts[(grid == grid[x][y - 1]).sum()] = 'up'
        if x < GAME_GRID_TILES[0]:
            counts[(grid == grid[x + 1][y]).sum()] = 'right'
        if y < GAME_GRID_TILES[1]:
            counts[(grid == grid[x][y + 1]).sum()] = 'down'
        swap_dir = counts[max(counts.keys())]
        if swap_dir == 'left':
            pyautogui.click(xpos - GAME_TILE_DIM[0], ypos, duration=0.2)
        elif swap_dir == 'up':
            pyautogui.click(xpos, ypos - GAME_TILE_DIM[1], duration=0.2)
        elif swap_dir == 'right':
            pyautogui.click(xpos + GAME_TILE_DIM[0], ypos, duration=0.2)
        else:
            pyautogui.click(xpos, ypos + GAME_TILE_DIM[1], duration=0.2)
        pyautogui.moveTo(Game_Region[0] - 100, Game_Region[1])
    else:
        raise RuntimeError('No valid match')


# Modified region screenshot function for convenience
def screenshot(filename, region):
    x1 = region[0]
    y1 = region[1]
    width = region[2]
    height = region[3]
    pyautogui.screenshot(filename, region=(x1, y1, width, height))


# Mass capture and export screenshots of a single tile
# Helps capture different frames on the same tile
def area_capture(x: int, y: int, region: Tuple[int, ...]):
    path = './images/capture/'
    if not os.path.exists(path):
        os.mkdir(path)
    cube_area = (Game_Region[0] + x * GAME_TILE_DIM[0], Game_Region[1] + y * GAME_TILE_DIM[1],
                 GAME_TILE_DIM[0], GAME_TILE_DIM[1])
    i = 0
    while i < 240:
        screenshot(path + str(i) + '.png', cube_area)
        print(i)
        i += 1


# Captures, divides, and exports a single image of every tile in the grid
def grid_capture():
    path = './images/capture/'
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(GAME_GRID_TILES[0]):
        for j in range(GAME_GRID_TILES[1]):
            tile = (Game_Region[0] + i * GAME_TILE_DIM[0], Game_Region[1] + j * GAME_TILE_DIM[1],
                    GAME_TILE_DIM[0], GAME_TILE_DIM[1])
            screenshot(path + str(i) + str(j) + '.png', tile)


# Creates neural net training data of gem images
# Most important factor was to train the net to identify tiles with animations
# Static tile images were unnecessary
def create_training_data():
    training_data = []
    class_num = 0
    for gem, value in GEM_MAP:
        color = gem.name.lower()
        flame_path = os.path.join(GEM_DIRECTORY, color + '_flame')
        for img_file in os.listdir(flame_path):
            img = cv2.imread(os.path.join(flame_path, img_file))
            training_data.append([img, class_num])
        star_path = os.path.join(GEM_DIRECTORY, color + '_star')
        for img_file in os.listdir(star_path):
            img = cv2.imread(os.path.join(star_path, img_file))
            training_data.append([img, class_num])
        class_num += 1
    hypercube_path = os.path.join(GEM_DIRECTORY, 'hypercube')
    for img_file in os.listdir(hypercube_path):
        img = cv2.imread(os.path.join(hypercube_path, img_file))
        training_data.append([img, class_num])
    random.shuffle(training_data)
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, GAME_TILE_DIM[0], GAME_TILE_DIM[1], 3)
    np.save('./neural_data/X.npy', X)
    np.save('./neural_data/y.npy', y)


# Creates and trains a neural net for identifying the gem within each tile
# Needs to recognize animated sprites
def build_neural_net(neural_name: str):
    X = np.load('./neural_data/X.npy')
    X = X / 255.0
    y = np.load('./neural_data/y.npy')
    model = Sequential()

    model.add(Conv2D(80, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(80))

    model.add(Dense(len(GEM_MAP)))
    model.add(Activation('softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, batch_size=40, epochs=5, validation_split=0.05)
    model.save(neural_name)


# Loads a saved neural net by name into the global region
def load_net(neural_name: str):
    global Gem_Neural_Net
    Gem_Neural_Net = load_model(neural_name)
    if Gem_Neural_Net is None:
        raise RuntimeError('Neural net failed to load')






if __name__ == '__main__':
    load_net('tile_identifier')
    sleep(1)

    # Finds the game window to start the game
    game_window = find_game()
    start_game(game_window)
    sleep(3)

    # # Calculate grid coordinates
    grid_x1 = game_window[0] + GAME_GRID_OFFSET[0]
    grid_y1 = game_window[1] + GAME_GRID_OFFSET[1]
    grid_x2 = grid_x1 + GAME_GRID_TILES[0] * GAME_TILE_DIM[0]
    grid_y2 = grid_y1 + GAME_GRID_TILES[1] * GAME_TILE_DIM[1]
    Game_Region = (grid_x1, grid_y1, grid_x2, grid_y2)
    pyautogui.moveTo(grid_x1 - 100, grid_y1, duration=0.25)

    # Begin playing the game
    play_loop()


    # Functions used to create data

    # area_capture(2, 4, grid_region)
    # grid_capture(grid_region)
    # play_loop(grid_region)

    # create_training_data()
    # build_neural_net()

