# Euler Charakteristik / Cech Complex

##### Imports #####
import numpy
import numpy as np 
import pylab as plt 
from scipy.spatial import Delaunay
import math 
from scipy import ndimage
from scipy.spatial import distance_matrix
import pandas as pd
from tabulate import tabulate
plt.close()

##### Eingabe #####
num_samples = 40
anzahl_punkte_pro_kreis = 4
epsilon = .1
radius = .1
teilkreis = 0

##### Funktionen #####
def punkte(xi,yi,radius,samples):    
    num_samples = samples
    theta = np.linspace(0, 2*np.pi, num_samples)
    
    r = radius*np.random.rand((num_samples))
    x, y = r * np.cos(theta)+xi, r * np.sin(theta)+yi
    return x,y

def dist(x,y):  
     dist = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)  
     return dist  
 
def abstaende(x,y,z):
    erster_abstand_punkte = dist(x,y)
    zweiter_abstand_punkte = dist(y,z)
    dritter_abstand_punkte = dist(x,z)
    maxi = max(erster_abstand_punkte,zweiter_abstand_punkte,dritter_abstand_punkte)
    return maxi

# Berechnung der Punkte in dem Kreisring
theta = np.linspace(0, 2*np.pi, num_samples)
a, b = 1 * np.cos(theta), 1 * np.sin(theta)

anzahl = num_samples - teilkreis

x = np.zeros([anzahl,anzahl_punkte_pro_kreis])
y = np.zeros([anzahl,anzahl_punkte_pro_kreis])

# Berechnung der Punkte
for i in range(anzahl):
    x_tmp , y_tmp =punkte(a[i],b[i],radius,anzahl_punkte_pro_kreis)
    x[i] = x_tmp
    y[i] = y_tmp

###########################################################################################################
# Das Folgende ist für das Simplizialkomplex
def naiveVR(points, epsilon):
   points = [numpy.array(x) for x in points]   
   vrComplex = [(x,y) for (x,y) in combinations(points, 2) if numpy.linalg.norm(x - y) < 2*epsilon]
   return numpy.array(vrComplex)

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = np.arange(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in (range(r)): # reversed
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
############################################################################################################
        
x = np.reshape(x,np.shape(x)[0]*np.shape(x)[1])
y = np.reshape(y,np.shape(y)[0]*np.shape(y)[1])

a = np.vstack((x.T,y.T)).T
points = a

#test = naiveVR(points, epsilon)
#plt.plot(test[:,1,0],test[:,1,1])
plt.plot(points[:,0],points[:,1],'.')

#from mogutda import SimplicialComplex
#SimplicialComplex.eulerCharacteristic()


# Distanzmatrix
df = pd.DataFrame(points)    
t = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).to_numpy()

# Epsilon Abstaende
k = np.where(t<epsilon,t,0)

# Punkte pro Epsilonball
kut = tri_upper_no_diag = np.triu(k,1)
kut[np.where(kut!=0)] = 1
anzahl_pe = sum(kut.T[:])+1
asdf = np.zeros(np.shape(k)[0]+1)
for i in range(np.shape(k)[0]):
    asdf[i+1] = np.count_nonzero(anzahl_pe == i+1)
    if np.count_nonzero(anzahl_pe == i+1) == 0:
        break
pb = asdf[0:i]

# Plot für Epsilonbaelle
print()
print('==============================')
print()
print('Epsilon = ',epsilon)
print()
gjk = np.vstack((np.arange(0,np.size(pb)).T,pb.T)).T
print(tabulate(gjk,headers=['Epsilon-Ball mit x Punkten', 'Anzahl']))














######### alles weitere ist erstmal egal ##########






#import cv2
#import numpy as np
#
##the image
#img = np.ones([10,10])
#img[3:7, 3:7] = 255
##print "Image:"
#print(img)
#
#blur_img = cv2.GaussianBlur(img, (3,3), 0)
#print
#print("Blur:")
#print(blur_img.astype(np.int))
#
#detected_edges = cv2.Canny(img.astype(np.uint8), 0, 0)
#print
#print("Canny without blur:")
#print(detected_edges.astype(np.int))
#
#detected_edges = cv2.Canny(blur_img.astype(np.uint8), 1, 255)
#print
#print("Canny with blur:")
#print(detected_edges.astype(np.int))
#
##from scipy import ndimage
#sx = ndimage.sobel(img, axis=0, mode='constant')
#sy = ndimage.sobel(img, axis=1, mode='constant')
#sob = np.hypot(sx, sy)
#print
#print("Sobel:")
#print(sob.astype(np.int))






#
## coding=utf-8
#
#import pygame, random
#from pygame.locals import *
#
#### begin: constants ###
#window_width = 640
#window_height = 480
#cell_size = 36 # the size of cells in pixels
#
## colors #
#white = (255, 255, 255)
#black = (0, 0, 0)
#gray = (100, 100, 100)
#red = (255, 0, 0)
#blue = (0, 0, 255)
#green = (0, 100, 0)
#light_green = (0, 255, 0)
#
#### end: constants ###
#
#
#### begin: global variables ###
#
## sizes #
#board_size = 0 # the number of cells in rows and columns
#board_width = cell_size * board_size + cell_size / 2 * (board_size - 1)
#board_height = cell_size * board_size
#xmargin = int((window_width - board_width) / 2)
#ymargin = int((window_height - board_height) / 2)
#
## players "
#red_player = "human"
#blue_player = "human"
#
## rule #
#rule = "normal" # or "reversed"
#
## buttons #
#
#buttons = []
#new_game_button = 0
#back_button = 0
#forward_button = 0
#order_button = 0
#order_displayed = False
#
## cells, moves and ... #
#moves = []
#moves_record = None
#cells = []
#real_cells = []
#empty_cells = []
#rects = []
#vertices = []
#edges = []
#
#restart = False # if this value is True, then the program exits the run_game loop and start a new game.
#repeat = True
#stats_dict = dict([])
#current_turn = red
#done_steps = 0
#total_steps_num = 100
#loops_num_each_step = 100
#
#### end: global variables ###
#
#
#### begin: main and start ###
#
#def main():
#    """
#    The main function. When the program runs, this function is called first.
#    """
#    global screen, font, big_font, order_font
#
#    pygame.init()
#    screen = pygame.display.set_mode((window_width, window_height), 0, 32)
#    pygame.display.set_caption("Euler Getter 2")
#
#    ### fonts ###
#    try:
#        font = pygame.font.SysFont("arial", 18)
#        order_font = pygame.font.SysFont("arial", 25)
#        big_font = pygame.font.SysFont("arial", 30)
#    except MemoryError:
#        font = pygame.font.font(pygame.font.get_default_font(), 18)
#        order_font = pygame.font.font(pygame.font.get_default_font(), 25)
#        big_font = pygame.font.font(pygame.font.get_default_font(), 30)
#        ##############
#
#    start()
#
#
#def quit(event):
#    """
#    quit the program if the exit button is pressed
#    """
#    if event.type == QUIT:
#        exit()
#
#
#def start():
#    """
#    start a new game
#    """
#    global board_size
#
#    choose_board_size()
#    choose_players()
#    choose_rule()
#    run_game()
#
#
#def choose_board_size():
#    """
#    choose-the-board-seize menu
#    """
#    global board_size, buttons
#
#    screen.fill(black)
#    choose_size_txt_surf = big_font.render("Choose the board size", True, white)
#    w = choose_size_txt_surf.get_width()
#    screen.blit(choose_size_txt_surf, (int((window_width - w) / 2), 150))
#
#    small_surf = big_font.render(" Small ", True, black, white)
#    medium_surf = big_font.render(" Medium ", True, black, white)
#    large_surf = big_font.render(" Large ", True, black, white)
#
#    sw = small_surf.get_width()
#    mw = medium_surf.get_width()
#    lw = large_surf.get_width()
#
#    interval = int((window_width - sw - mw - lw) / 4)
#    screen.blit(small_surf, (interval, 300))
#    screen.blit(medium_surf, (sw + 2 * interval, 300))
#    screen.blit(large_surf, (sw + mw + 3 * interval, 300))
#
#    small_button = small_surf.get_rect()
#    medium_button = medium_surf.get_rect()
#    large_button = large_surf.get_rect()
#
#    small_button.topleft = (interval, 300)
#    medium_button.topleft = (sw + 2 * interval, 300)
#    large_button.topleft = (sw + mw + 3 * interval, 300)
#
#    buttons = [small_button, medium_button, large_button]
#
#    pygame.display.update()
#
#    button = get_pressed_button()
#    while button == None:
#        button = get_pressed_button()
#        if button == small_button:
#            board_size = 6
#        elif button == medium_button:
#            board_size = 8
#        elif button == large_button:
#            board_size = 10
#
#
#def choose_players():
#    """
#    choose the players menu
#    """
#    global red_player, blue_player, buttons
#
#    screen.fill(black)
#
#    ### choose the first (red) player
#    choose_red_txt_surf = big_font.render("Choose the first (red) player", True, white)
#    w = choose_red_txt_surf.get_width()
#    screen.blit(choose_red_txt_surf, (int((window_width - w) / 2), 150))
#
#    human_surf = big_font.render(" Human ", True, black, white)
#    computer_surf = big_font.render(" Computer ", True, black, white)
#
#    hw = human_surf.get_width()
#    cw = computer_surf.get_width()
#
#    interval = int((window_width - hw - cw ) / 3)
#    screen.blit(human_surf, (interval, 300))
#    screen.blit(computer_surf, (hw + 2 * interval, 300))
#
#    human_button = human_surf.get_rect()
#    computer_button = computer_surf.get_rect()
#
#    human_button.topleft = (interval, 300)
#    computer_button.topleft = (hw + 2 * interval, 300)
#
#    buttons = [human_button, computer_button]
#
#    pygame.display.update()
#
#    button = get_pressed_button()
#    while button == None:
#        button = get_pressed_button()
#        if button == human_button:
#            red_player = "human"
#        elif button == computer_button:
#            red_player = "computer"
#
#
#    ### choose the second (blue) player
#    screen.fill(black)
#
#    choose_blue_txt_surf = big_font.render("Choose the second (blue) player", True, white)
#    w = choose_blue_txt_surf.get_width()
#    screen.blit(choose_blue_txt_surf, (int((window_width - w) / 2), 150))
#
#    screen.blit(human_surf, (interval, 300))
#    screen.blit(computer_surf, (hw + 2 * interval, 300))
#
#    pygame.display.update()
#
#    button = get_pressed_button()
#    while button == None:
#        button = get_pressed_button()
#        if button == human_button:
#            blue_player = "human"
#        elif button == computer_button:
#            blue_player = "computer"
#
#
#def choose_rule():
#    """
#    choose the normal or reversed rule. in the reverse rule, the player with negative euler char. wins.
#    """
#    global rule, buttons
#
#    screen.fill(black)
#
#    choose_rule_surf = big_font.render("Choose the rule", True, white)
#    w = choose_rule_surf.get_width()
#    screen.blit(choose_rule_surf, (int((window_width - w) / 2), 150))
#
#    normal_surf = big_font.render(" Normal ", True, black, white)
#    reversed_surf = big_font.render(" Reversed ", True, black, white)
#
#    nw = normal_surf.get_width()
#    rw = reversed_surf.get_width()
#
#    interval = int((window_width - nw - rw ) / 3)
#    screen.blit(normal_surf, (interval, 300))
#    screen.blit(reversed_surf, (nw + 2 * interval, 300))
#
#    normal_button = normal_surf.get_rect()
#    reversed_button = reversed_surf.get_rect()
#
#    normal_button.topleft = (interval, 300)
#    reversed_button.topleft = (nw + 2 * interval, 300)
#
#    buttons = [normal_button, reversed_button]
#
#    pygame.display.update()
#
#    button = get_pressed_button()
#    while button == None:
#        button = get_pressed_button()
#        if button == normal_button:
#            rule = "normal"
#        elif button == reversed_button:
#            rule = "reversed"
#
#### end: main and start ###
#
#
#### begin: handling buttons ###
#
#def get_pressed_button():
#    """
#    wait for a button to be pressed
#    """
#    for event in pygame.event.get():
#        quit(event)
#
#        if event.type != MOUSEBUTTONDOWN:
#            return None
#        else:
#            mouse_position = event.pos
#            for button in buttons:
#                if button.collidepoint(mouse_position):
#                    return button
#            else:
#                return None
#
#
#def get_pressed_button2(event):
#    """
#    Given an event, returns a pressed button if any.
#    """
#    if event.type != MOUSEBUTTONDOWN:
#        return None
#    else:
#        mouse_position = event.pos
#        for button in buttons:
#            if button.collidepoint(mouse_position):
#                return button
#        else:
#            return None
#
#### end: handling buttons ###
#
#
#### begin: board, cells and moves ###
#
#def new_board():
#    """
#    initializes the board data
#    """
#    global cells, real_cells, rects, xmargin, ymargin, vertices, vertex_dict, edge_dict, moves
#
#    board_width = cell_size * board_size + cell_size / 2 * (board_size - 1)
#    board_height = cell_size * board_size
#    xmargin = int((window_width - board_width) / 2)
#    ymargin = int((window_height - board_height) / 2)
#    cells = [(i, j) for i in range(board_size) for j in range(board_size)]
#    rects = dict([[cell, get_rect(cell)] for cell in cells])
#    real_cells = [cell for cell in cells if is_real(cell)]
#    new_vertices(cells)
#    new_edges(cells)
#    vertex_dict = dict([(cell, get_vertices(cell)) for cell in cells])
#    edge_dict = dict([(cell, get_edges(cell)) for cell in cells])
#    moves = []
#
#
#def get_topleft(cell):
#    """
#    return the coordinates of the topleft corner of the given cell
#    """
#    i, j = cell
#    x = xmargin + j * cell_size + int(cell_size / 2 * (board_size - 1 - i))
#    y = ymargin + i * cell_size
#    return x, y
#
#
#def get_rect(cell):
#    """
#    returns the Rect object of the cell
#    """
#    return Rect(get_topleft(cell), (cell_size, cell_size))
#
#
#def new_vertices(cells):
#    """
#    initializes the list of all vertices
#    """
#    global vertices
#
#    for cell in cells:
#        i, j = cell
#        if i == 0 and j == 0:
#            pass
#        elif i == 0 and j > 0:
#            vertices.append(get_rect(cell).bottomleft)
#        elif 0 < i < board_size - 1 and j > 0:
#            vertices.append(get_rect(cell).topleft)
#            vertices.append(get_rect(cell).bottomleft)
#        elif 0 < i < board_size - 1 and j == 0:
#            pass
#        elif i == board_size - 1 and j == 0:
#            pass
#        elif i == board_size - 1 and j > 0:
#            vertices.append(get_rect(cell).topleft)
#
#
#def topmidleft(rect):
#    x, y = rect.midtop
#    x1, y1 = rect.topleft
#    return int((x + x1) / 2), y
#
#
#def topmidright(rect):
#    x, y = rect.midtop
#    x1, y1 = rect.topright
#    return int((x + x1) / 2), y
#
#
#def new_edges(cells):
#    """
#    initializes the list of all edges. Edges are represented by the coordinates of their midpoints
#    """
#    global edges
#
#    for cell in cells:
#        i, j = cell
#        if i == 0 and j == 0:
#            pass
#        elif i == 0 and j > 0:
#            edges.append(get_rect(cell).midleft)
#        elif 0 < i < board_size - 1 and j == 0:
#            edges.append(topmidright(get_rect(cell)))
#        elif 0 < i < board_size - 1 and 0 < j < board_size - 1:
#            edges.append(topmidright(get_rect(cell)))
#            edges.append(topmidleft(get_rect(cell)))
#            edges.append(get_rect(cell).midleft)
#        elif 0 < i < board_size - 1 and j == board_size - 1:
#            edges.append(topmidleft(get_rect(cell)))
#            edges.append(get_rect(cell).midleft)
#        elif i == board_size - 1 and j == 0:
#            edges.append(topmidright(get_rect(cell)))
#        elif i == board_size - 1 and 0 < j < board_size - 1:
#            edges.append(topmidright(get_rect(cell)))
#            edges.append(topmidleft(get_rect(cell)))
#        else:
#            edges.append(topmidleft(get_rect(cell)))
#
#
#def cells_of(color, moves):
#    """
#    extract the moves of the given color
#    """
#    if color == red:
#        return moves[0::2]
#    if color == blue:
#        return moves[1::2]
#
#
#def is_real(cell):
#    """
#    we call one from the pair of partner cells "real" and the other "shadow".
#    """
#    if cell[0] < board_size - 1 and cell[1] < board_size - 1:
#        return True
#    elif cell == (0, board_size - 1):
#        return True
#    else:
#        return False
#
#
#def empty_cells(moves):
#    """
#    return the set of empty (real) cells
#    """
#    return set(real_cells).difference(moves)
#
#
#def on_boundary(cell):
#    """
#    check if the given cell is on the boundary of the board
#    """
#    if cell[0] in [0, board_size - 1] or cell[1] in [0, board_size - 1]:
#        return True
#    else:
#        return False
#
#
#def get_partner(cell):
#    """
#    return the partner cell if the cell is on the boundary, and None otherwise.
#    """
#    x, y = cell
#    if on_boundary(cell):
#        return board_size - 1 - x, board_size - 1 - y
#    else:
#        return None
#
#### end: board, cells and moves ###
#
#
#### begin: computing euler characteristics ###
#
#def get_big_rect(cell):
#    """
#    returns a slightly bigger Rect object of the cell. This is to capture vetices and edges.
#    """
#    x, y = get_topleft(cell)
#    return Rect((x - 2, y - 2), (cell_size + 4, cell_size + 4))
#
#def get_vertices(cell):
#    """
#    returns the set of the vertices of the cell
#    """
#    rect = get_big_rect(cell)
#    vs = [vertex for vertex in vertices if rect.collidepoint(vertex)]
#    if on_boundary(cell):
#        rect1 = get_big_rect(get_partner(cell))
#        vs1 = [vertex for vertex in vertices if rect1.collidepoint(vertex)]
#        vs.extend(vs1)
#    return set(vs)
#
#def get_edges(cell):
#    """
#    returns the set of the edges of the cell
#    """
#    rect = get_big_rect(cell)
#    es = [edge for edge in edges if rect.collidepoint(edge)]
#    if on_boundary(cell):
#        rect1 = get_big_rect(get_partner(cell))
#        es1 = [edge for edge in edges if rect1.collidepoint(edge)]
#        es.extend(es1)
#    return set(es)
#
#def euler_char(cells):
#    """
#    returns the Euler characteristic of the union of the cells.
#    """
#    vs = set([]) # a variable for the set of vertices
#    es = set([]) # a variable for the set of edges
#    c = len(cells)
#    for cell in cells:
#        vs = vs.union(vertex_dict[cell])
#        es = es.union(edge_dict[cell])
#    v = len(vs)
#    e = len(es)
#    return v - e + c
#
#### end: computing euler characteristics ###
#
#
#### begin: game process ###
#
#def run_game():
#    """
#    runs a single game
#    """
#    global restart
#
#    new_board()
#
#    restart = False
#
#    while not restart:
#        draw_all()
#
#        if game_finished():
#            human_turn()
#
#        elif get_player(turn()) == "human":
#            human_turn()
#        else:
#            computer_turn()
#
#    if restart:
#        start()
#
#
#def human_turn():
#    global moves, moves_record, repeat
#
#    repeat = True
#
#    while repeat:
#        for event in pygame.event.get():
#            quit(event)
#
#            button_action(event)
#
#            if event.type == MOUSEBUTTONDOWN:
#                cell = get_pointed_cell()
#                if cell != None:
#                    if not is_real(cell):
#                        cell = get_partner(cell)
#
#                    if cell not in moves:
#                        moves.append(cell)
#                        moves_record = None
#                        repeat = False
#
#            draw_all()
#
#
#def computer_turn():
#    global moves, moves_record, repeat, current_turn, done_steps, empty_cells
#
#    current_turn = turn()
#    empty_cells = list(set(real_cells).difference(moves))
#    new_stats_dict()
#    done_steps = 0
#
#    while done_steps < total_steps_num:
#        for event in pygame.event.get():
#            quit(event)
#
#            button_action(event)
#
#        computer_thinking_step()
#
#        draw_all()
#
#    moves.append(get_best_move())
#    moves_record = None
#
#    draw_all()
#
#
#### begin: AI (Monte Carlo) ###
#def new_stats_dict():
#    """
#    initializes the stats dictionary data.
#    """
#    global stats_dict
#
#    keys = [(cell, "wins") for cell in empty_cells] + [(cell, "losses") for cell in empty_cells]
#    stats_dict = dict.fromkeys(keys, 0)
#
#def computer_thinking_step():
#    """
#    runs a thinking step which includes "loops_num_each_step" ramdom trials.
#    """
#    global done_steps, empty_cells
#
#    for i in range(loops_num_each_step):
#        random.shuffle(empty_cells)
#        update_stats(empty_cells)
#
#    done_steps += 1
#
#def update_stats(filled_empties):
#    """
#    updates the stats dictionary by a random trial filling empty cells.
#    """
#    global stats_dict
#    filled_moves = moves + filled_empties
#
#    if get_winner(filled_moves) == red: # if red wins,
#        if current_turn == red:
#            red_empties = filled_empties[0::2]
#            for cell in red_empties:
#                stats_dict[(cell,"wins")] += 1
#        else:
#            blue_empties = filled_empties[0::2]
#            for cell in blue_empties:
#                stats_dict[(cell, "losses")] += 1
#    else: # if blue wins,
#        if current_turn == red:
#            red_empties = filled_empties[0::2]
#            for cell in red_empties:
#                stats_dict[(cell, "losses")] += 1
#        else:
#            blue_empties = filled_empties[0::2]
#            for cell in blue_empties:
#                stats_dict[(cell,"wins")] += 1
#
#
#def get_best_move():
#    """
#    From the stats dictionary, computes the next move which has the highest winning ratio and returns it.
#    """
#    ratio_dict = dict([])
#
#    for cell in empty_cells:
#        ws = stats_dict[(cell, "wins")]
#        ls = stats_dict[(cell,"losses")]
#        if ws == 0 and ls == 0:
#            ratio_dict[cell] = 0
#        else:
#            ratio_dict[cell] = float(ws) / (ws + ls)
#
#    list_pairs = ratio_dict.items() # the list of empty cells and their winning ratios
#    max_cell, max_ratio = max(list_pairs, key = lambda x : x[1])
#
#    return max_cell
#
#### end: AI (Monte Carlo) ###
#
#def turn():
#    if len(moves) % 2 == 0:
#        return red
#    else:
#        return blue
#
#
#def button_action(event):
#    """
#    During the game, this function works when some buttons is pressed.
#    """
#    global restart, repeat, moves, moves_record, order_displayed
#    pressed_button = get_pressed_button2(event)
#
#    if pressed_button == new_game_button:
#        if really_restart():
#            repeat = False
#            restart = True
#        else:
#            repeat = False
#    elif pressed_button == back_button:
#        if moves:
#            if moves_record == None:
#                moves_record = moves[:] # shallow copy
#            moves.pop()
#        repeat = False
#
#    elif pressed_button == forward_button:
#        if moves_record != None:
#            l = len(moves)
#            moves = moves_record[0:l + 1]
#            if moves == moves_record:
#                moves_record = None
#            repeat = False
#    elif pressed_button == order_button:
#        order_displayed = not order_displayed
#        repeat = False
#
#
#def really_restart():
#    """
#    asks the player if he really quits the current game and start a new one.
#    """
#    global buttons
#
#    center_surf = pygame.Surface((500, 300))
#    center_surf.fill(green)
#
#    really_surf = big_font.render("Do you really start a new game?", True, white)
#    x = int((500 - really_surf.get_width()) / 2)
#
#    center_surf.blit(really_surf, (x, 100))
#
#    yes_surf = big_font.render(" Yes ", True, black, white)
#    cancel_surf = big_font.render(" Cancel ", True, black, white)
#
#    interval = int((500 - yes_surf.get_width() - cancel_surf.get_width()) / 3)
#
#    center_surf.blit(yes_surf, (interval, 200))
#    center_surf.blit(cancel_surf, (yes_surf.get_width() + 2 * interval, 200))
#
#    screen.blit(center_surf, (int((window_width - 500) / 2), int((window_height - 300) / 2)))
#
#    pygame.display.update()
#
#    yes_button = yes_surf.get_rect()
#    cancel_button = cancel_surf.get_rect()
#
#    yes_button.topleft = (int((window_width - 500) / 2) + interval, int((window_height - 300) / 2) + 200  )
#    cancel_button.topleft = (
#    int((window_width - 500) / 2) + yes_surf.get_width() + 2 * interval, int((window_height - 300) / 2) + 200 )
#
#    buttons = [yes_button, cancel_button]
#
#    button = None
#
#    while button == None:
#        button = get_pressed_button()
#        if button == yes_button:
#            return True
#        elif button == cancel_button:
#            return False
#
#
#def get_player(red_or_blue):
#    """
#    Given red or blue, returns "human" or "computer"
#    """
#    if red_or_blue == red:
#        return red_player
#    else:
#        return blue_player
#
#
#def get_text(red_or_blue):
#    if red_or_blue == red:
#        return "Red"
#    elif red_or_blue == blue:
#        return "Blue"
#    else:
#        return None
#
#
#def get_color(moves, cell):
#    """
#    given a cell and the list of moves, return the color of the cell if it is red or blue, and white if it is empty.
#    """
#    reds = cells_of(red, moves)
#    blues = cells_of(blue, moves)
#
#    cell1 = cell
#    if not is_real(cell):
#        cell1 = get_partner(cell)
#
#    if cell1 in reds:
#        return red
#    elif cell1 in blues:
#        return blue
#    else:
#        return white
#
#
#def get_order(moves, cell):
#    """
#    returns the order number of the cell if the cell is in moves, and None otherwise.
#    """
#    cell1 = cell
#
#    if not is_real(cell):
#        cell1 = get_partner(cell)
#
#    if cell1 in moves:
#        return moves.index(cell1)
#    else:
#        return None
#### end: game process ###
#
#
#### begin: drawing ###
#
#def draw_cell(cell, color):
#    pygame.draw.rect(screen, color, rects[cell])
#    pygame.draw.rect(screen, black, rects[cell], 1)
#
#
#def draw_number_on_cell(cell, number):
#    if number == None:
#        pass
#    else:
#        number_surf = order_font.render(str(number), True, white)
#        x, y = get_topleft(cell)
#        screen.blit(number_surf, (x + 3, y + 3))
#
#
#def draw_board():
#    global buttons, new_game_button
#
#    for cell in cells:
#        draw_cell(cell, get_color(moves, cell))
#
#    if order_displayed:
#        for cell in cells:
#            draw_number_on_cell(cell, get_order(moves, cell))
#
#
#def draw_top_line():
#    """
#    draw "new game", "back", "forward", and "displayer the move order" buttons.
#    """
#    global buttons, new_game_button, back_button, forward_button, order_button
#
#    new_game_surf = font.render(" New game ", True, black, white)
#
#    if not moves:
#        back_surf = font.render(" Back ", True, black, gray)
#    else:
#        back_surf = font.render(" Back ", True, black, white)
#
#    if moves_record == None:
#        forward_surf = font.render(" Forward ", True, black, gray)
#    else:
#        forward_surf = font.render(" Forward ", True, black, white)
#
#    if order_displayed:
#        order_surf = font.render(" Hide the move order ", True, black, white)
#    else:
#        order_surf = font.render(" Display the move order ", True, black, white)
#
#    nw = new_game_surf.get_width()
#    bw = back_surf.get_width()
#    fw = forward_surf.get_width()
#    ow = order_surf.get_width()
#
#    interval = int((window_width - nw - bw - fw - ow) / 5)
#
#    screen.blit(new_game_surf, (interval, 10))
#    screen.blit(back_surf, (nw + 2 * interval, 10))
#    screen.blit(forward_surf, (nw + bw + 3 * interval, 10))
#    screen.blit(order_surf, (nw + bw + fw + 4 * interval, 10))
#
#    new_game_button = new_game_surf.get_rect()
#    back_button = back_surf.get_rect()
#    forward_button = forward_surf.get_rect()
#    order_button = order_surf.get_rect()
#
#    new_game_button.topleft = (interval, 10)
#    back_button.topleft = (nw + 2 * interval, 10)
#    forward_button.topleft = (nw + bw + 3 * interval, 10)
#    order_button.topleft = (nw + bw + fw + 4 * interval, 10)
#
#    buttons = [new_game_button, back_button, forward_button, order_button]
#
#
#def draw_bottom_line():
#    """
#    draw the turn, the Euler characteristics and so on.
#    """
#    red_euler = euler_char(cells_of(red, moves))
#    blue_euler = euler_char(cells_of(blue, moves))
#
#    last_text = "%s's turn" % get_text(turn())
#
#    if game_finished():
#        last_text = "%s won!!" % get_text(get_winner(moves))
#
#    bottom_text = "Rule: %s  ||  Red (%s): %d  ||  Blue (%s): %d  ||  %s" % (
#    rule, red_player, red_euler, blue_player, blue_euler, last_text)
#    bottom_text_surf = font.render(bottom_text, True, white)
#    w = bottom_text_surf.get_width()
#
#    screen.blit(bottom_text_surf, (int((window_width - w) / 2), window_height - 40))
#
#def get_winner(total_moves):
#    red_euler = euler_char(cells_of(red, total_moves))
#    if red_euler > 0 and rule == "normal":
#        return red
#    elif red_euler <= 0 and rule == "reversed":
#        return red
#    else:
#        return blue
#
#
#def game_finished():
#    """
#    checks if the game is finished and returns a Boolean value
#    """
#    if len(moves) == len(real_cells):
#        return True
#    else:
#        return False
#
#
#def draw_pointed_cell():
#    pointed_cell = get_pointed_cell()
#
#    if pointed_cell != None:
#        pygame.draw.rect(screen, light_green, rects[pointed_cell], 4)
#        if on_boundary(pointed_cell):
#            pygame.draw.rect(screen, light_green, rects[get_partner(pointed_cell)], 4)
#
#
#def get_pointed_cell():
#    mouse_position = pygame.mouse.get_pos()
#    pointed_cells = [cell for cell in cells if rects[cell].collidepoint(mouse_position)]
#    pointed_cell = None
#
#    if pointed_cells != []:
#        pointed_cell = pointed_cells[0]
#
#    return pointed_cell
#
#
#def draw_all():
#    screen.fill(black)
#    draw_board()
#    draw_pointed_cell()
#    draw_top_line()
#    draw_bottom_line()
#    pygame.display.update()
#
#### end: drawing ###
#
#
#### run the program ###
#if __name__ == "__main__":
#    main()
#
#
#
#
#
#
#
