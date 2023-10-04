import AlphaZero.GI
using StaticArrays

const BOARD_SIZE = 9
const NUM_COLS = BOARD_SIZE
const NUM_ROWS = BOARD_SIZE
const NUM_POSITIONS = BOARD_SIZE ^ 2

const Player = Int
const WHITE = -1
const EMPTY = 0
const BLACK = 1
const FILL = 2
const KO = 3
const UNKNOWN = 4

const Cell = Player
# const Board = MMatrix{NUM_ROWS, NUM_COLS, Cell, NUM_POSITIONS} # SMatrix{NUM_COLS, NUM_ROWS, Cell, NUM_POSITIONS}
# const INITIAL_BOARD = @MMatrix zeros(Cell, NUM_ROWS, NUM_COLS)
const Board = SMatrix{NUM_COLS, NUM_ROWS, Cell, NUM_POSITIONS}
const INITIAL_BOARD = @SMatrix zeros(Cell, NUM_COLS, NUM_ROWS)
const NUM_PLANES = 17
# const EMPTY_BOARD = @MMatrix zeros(Cell, NUM_ROWS, NUM_COLS)
# const ALL_COORDS = [(i, j) for i = 1:NUM_ROWS for j = 1:NUM_COLS]
# check_bounds(c) = 1 <= c[1] <= BOARD_SIZE && 1 <= c[2] <= BOARD_SIZE
# const NEIGHBORS = Dict((x, y) => filter(k->check_bounds(k),[(x+1, y), (x-1, y), (x, y+1), (x, y-1)]) for (x, y) in ALL_COORDS)
# const DIAGONALS = Dict((x, y) => filter(k->check_bounds(k),[(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]) for (x, y) in ALL_COORDS)
# const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=BLACK, finished=false, winner = 0x00, amask = trues(NUM_POSITIONS), boardSize=BOARD_SIZE, planes=8, emptyBoard=EMPTY_BOARD, neighbors=NEIGHBORS,diagonals=DIAGONALS)
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

# TODO: we could have the game parametrized by grid size.
struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  boardSize::Int
  planes::Int
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  amask :: Vector{Bool} # actions mask

#   action_space::Int

#   max_action_space::Int
  emptyBoard::Board
  neighbors::Dict{NTuple{2, Int}, Array{NTuple{2, Int}, 1}}
  diagonals::Dict{NTuple{2, Int}, Array{NTuple{2, Int}, 1}}

  function GameEnv(board::Board = INITIAL_BOARD, curplayer::Cell = BLACK)
    N = BOARD_SIZE
    L = N * N
    ALL_COORDS_N = [(i, j) for i = 1:N for j = 1:N]
    EMPTY_BOARD_N = SMatrix{N, N, Cell, L}(zeros(Int8, N, N))
    planes = NUM_PLANES
    @assert planes % 2 == 1
    planes = (planes - 1) ÷ 2
    # current_player = BLACK
    finished = false
    winner = 0x00
    amask = trues(NUM_POSITIONS)
    check_bounds(c) = 1 <= c[1] <= N && 1 <= c[2] <= N

    NEIGHBORS_N = Dict((x, y) => filter(k->check_bounds(k),
                              [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]) for (x, y) in ALL_COORDS_N)
    DIAGONALS_N = Dict((x, y) => filter(k->check_bounds(k),
                              [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]) for (x, y) in ALL_COORDS_N)
    new(N, planes, board, curplayer, finished, winner, amask, EMPTY_BOARD_N, NEIGHBORS_N, DIAGONALS_N)
  end

end

include("board.jl")

GI.init(::GameSpec, state=INITIAL_STATE) = GameEnv(state.board, state.curplayer) #GameEnv(state.boardSize, state.planes, state.board, state.curplayer, state.finished, state.winner, state.amask, state.emptyBoard, NEIGHBORS, DIAGONALS)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = true

function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  update_actions_mask!(g)
  any(g.amask) || (g.finished = true)
#   g.finished = state.find_reached
  # g.winner = state.winner
  # g.amask = state.amask
  # g.boardSize = state.boardSize
  # g.planes = state.planes
  # g.emptyBoard = state.emptyBoard
  # g.neighbors = state.neighbors
  # g.diagonals = state.diagonals
end

#####
##### Defining winning conditions
#####

pos_of_xy((x, y)) = (y - 1) * BOARD_SIZE + (x - 1) + 1

xy_of_pos(pos) = ((pos - 1) % BOARD_SIZE + 1, (pos - 1) ÷ BOARD_SIZE + 1)

# Position(env::GameEnv; args...) = GoPosition(env; args...)

function has_won(g::GameEnv, player)
  avail_actions = GI.available_actions(g)
  pos = GoPosition(g; komi = -7.5, to_play = g.curplayer)
  points = score(pos)  
  won = isempty(avail_actions) && ((points > 0  && player == BLACK) || (points < 0 && player == WHITE))  
  return won
end

#####
##### Game API
#####

const ACTIONS = collect(1:NUM_POSITIONS)

getmask(c) = c == EMPTY

GI.actions(::GameSpec) = ACTIONS

function update_actions_mask!(g::GameEnv)
  g.amask = map(ACTIONS) do pos
    # g.board[pos] == EMPTY
    position = GoPosition(g; komi = -7.5, to_play = g.curplayer)
    c = xy_of_pos(pos)
    valid = is_move_legal(position, c)
    if valid
      try
        play_move!(position, c; color = position.to_play)
        valid = true
      catch e
        # print("exception:", e, "\n\n")
        if isa(e, IllegalMove)
          valid = false
        end
      end
    end
    return valid
  end
  # len = length(g.amask)
  # for p in collect(1:len)
  #   flag = (g.board[p] == EMPTY)
  #   g.amask[p] = flag
  # end
end

function GI.actions_mask(g::GameEnv)
  update_actions_mask!(g)
  return g.amask
end

GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

function terminal_white_reward(g::GameEnv)
  has_won(g, WHITE) && return -1.
  has_won(g, BLACK) && return 1.
  avail_actions = GI.available_actions(g)
  # print("g.amask=", g.amask, " avail_actions=", avail_actions, "\n\n")
  # if isempty(avail_actions)
  #   print("g.amask=", g.amask, ", avail_actions=", avail_actions, "\n\n")
  # else 
  #   print("g.amask=", g.amask, ", avail_actions_count=", length(avail_actions), "\n\n")
  # end
  isempty(avail_actions) && return 0.
  return nothing
end

function GI.game_terminated(g::GameEnv)
  update_actions_mask!(g)
  twr = terminal_white_reward(g)
  # print("GI.game_terminated: terminal_white_reward=", twr, "\n\n")
  !isnothing(twr)
end

function GI.white_reward(g::GameEnv)
  z = terminal_white_reward(g)
  return isnothing(z) ? 0. : z
end

function GI.play!(g::GameEnv, pos)
  # g.board = setindex(g.board, g.curplayer, pos)
  # g.board[pos] = g.curplayer
  # update_actions_mask!(g)
  position = GoPosition(g; komi = -7.5, to_play = g.curplayer)
  c = xy_of_pos(pos)
  newPosition = play_move!(position, c; color = position.to_play)
  len = length(newPosition.board)
  for i in collect(1:len)
    # g.board[i] = newPosition.board[i]
    (col, row) = xy_of_pos(i)
    player = newPosition.board[col, row]
    g.board = setindex(g.board, player, col, row)
  end
  # update_actions_mask!(g)
  g.curplayer = -g.curplayer
  # p = g.curplayer
  # m = g.amask
  # println("curplayer=", g.curplayer, " board=", g.board, "\n\n")
end

#####
##### Simple heuristic for minmax
#####

function score_for(g::GameEnv, player) ::Float64
  pos = GoPosition(g; komi = -7.5, to_play = g.curplayer)
  estimate = score(pos)
  if player == WHITE
    estimate = -estimate
  end
  return estimate
end

function heuristic_value_for(g::GameEnv, player)
  return score_for(g, player)
end

function GI.heuristic_value(g::GameEnv)
  mine = heuristic_value_for(g, g.curplayer)
  yours = heuristic_value_for(g, -g.curplayer)
  return mine - yours
end

#####
##### Machine Learning API
#####

function flip_colors(board)
  flip(cell) = cell == WHITE ? BLACK : (cell == BLACK ? WHITE : cell)
  # Inference fails when using `map`
  return @SVector Cell[flip(board[i]) for i in 1:NUM_POSITIONS]
end

# Vectorized representation: 3x3x3 array
# Channels: free, white, black
# The board is represented from the perspective of white
# (as if white were to play next)
function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    board[pos_of_xy((x, y))] == c
    for x in 1:BOARD_SIZE,
        y in 1:BOARD_SIZE,
        c in [nothing, WHITE, BLACK]]
end

#####
##### Symmetries
#####

function generate_dihedral_symmetries()
  N = BOARD_SIZE
  rot((x, y)) = (y, N - x + 1) # 90° rotation
  flip((x, y)) = (x, N - y + 1) # flip along vertical axis
  ap(f) = p -> pos_of_xy(f(xy_of_pos(p)))
  sym(f) = map(ap(f), collect(1:NUM_POSITIONS))
  rot2 = rot ∘ rot
  rot3 = rot2 ∘ rot
  return [
    sym(rot), sym(rot2), sym(rot3),
    sym(flip), sym(flip ∘ rot), sym(flip ∘ rot2), sym(flip ∘ rot3)]
end

const SYMMETRIES = generate_dihedral_symmetries()

function GI.symmetries(::GameSpec, s)
  return [
    ((board=Board(s.board[sym]), curplayer=s.curplayer), sym)
    for sym in SYMMETRIES]
end

#####
##### Interaction API
#####

function GI.action_string(::GameSpec, a)
  x, y = xy_of_pos(a)
  string(Char(Int('A') + x - 1)) * string(Char(Int('A') + y - 1))
end

function GI.parse_action(::GameSpec, str)
  length(str) == 2 || (return nothing)
  x = Int(uppercase(str[1])) - Int('A') + 1
  y = Int(uppercase(str[2])) - Int('A') + 1
  n = pos_of_xy((x, y))
  (0 < n < NUM_POSITIONS + 1) ? n : nothing
end

function read_board(::GameSpec)
  n = BOARD_SIZE
  str = reduce(*, ((readline() * repeat(" ", n))[1:n] for i in 1:n))
  white = ['w', 'r', 'o']
  black = ['b', 'b', 'x']
  function cell(i)
    if (str[i] ∈ white) WHITE
    elseif (str[i] ∈ black) BLACK
    else nothing end
  end
  @SVector [cell(i) for i in 1:NUM_POSITIONS]
end

function GI.read_state(::GameSpec)
  b = read_board(GameSpec())
  nw = count(==(WHITE), b)
  nb = count(==(BLACK), b)
  if nw == nb
    return (board=b, curplayer=WHITE)
  elseif nw == nb + 1
    return (board=b, curplayer=BLACK)
  else
    return nothing
  end
end

using Crayons

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "White" : "Black"
player_mark(p)  = p == WHITE ? "o" : p == BLACK ? "x" : " "

function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname, " plays:", crayon"reset", "\n\n")
  for y in 1:BOARD_SIZE
    for x in 1:BOARD_SIZE
      pos = pos_of_xy((x, y))
      c = g.board[pos]
      if isnothing(c) || c == EMPTY
        print(".")
      elseif c == KO
        print("*")
      elseif c == FILL
        print("#")
      else
        print(player_color(c), player_mark(c), crayon"reset")
      end
      print(" ")
    end
    if with_position_names
      print(" | ")
      for x in 1:BOARD_SIZE
        print(GI.action_string(GI.spec(g), pos_of_xy((x, y))), " ")
      end
    end
    print("\n")
  end
  botmargin && print("\n")
end
