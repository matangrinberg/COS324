import numpy as np
import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

c = 0
class Learner:

  def __init__(self):
    self.last_state  = { 'score': 0, 'tree': {'dist': 0, 'top': 0, 'bot': 0}, 'monkey': {'vel': 0, 'top': 0, 'bot': 0}}
    self.last_action = 0
    self.last_reward = 0
    self.q           = np.zeros([8, 16, 8, 5, 2])
    self.rate        = np.ones([8, 16, 8, 5, 2])
    self.gamma       = 0.3
    self.totscore    = 0
    self.hscore      = 0
    self.eps         = 1

  # def eps_incr(self):
  #   self.eps = 1/np.power((np.power(1/self.eps, 1/1.5) +1), 1.5)
  #   # print(self.eps)

  def total_score(self):
    return self.totscore

  def high_score(self):
    return self.hscore

  def reset(self):
    self.last_state = {'score': 0, 'tree': {'dist': 0, 'top': 0, 'bot': 0}, 'monkey': {'vel': 0, 'top': 0, 'bot': 0}}
    self.last_action = 0
    self.last_reward = 0

  def action_callback(self, state):
    """Implement this function to learn things and take actions.
    Return 0 if you don't want to jump and 1 if you do."""
    # You might do some learning here based on the current state and the last state.

    # we ignore monkey top because it does not provide unique info (monkey bot suffices)
    # scale down from 600 x 400 to 600/scalefactor x 400/scalefactor
    # vel 40 to -40
    scale_factor = 50
    vel_factor = 20
    tree_factor = scale_factor
    dist_factor = 1.5 * scale_factor

    # Old State
    old_tree       = self.last_state.get('tree')
    old_tree_dist  = int(np.floor(old_tree.get('dist')/dist_factor))
    old_tree_bot   = int(np.floor(old_tree.get('bot')/(tree_factor/2)))
    old_monkey     = self.last_state.get('monkey')
    old_monkey_bot = int(np.floor(old_monkey.get('bot') / scale_factor))
    old_monkey_vel = int(np.floor(old_monkey.get('vel')/vel_factor))+3


    # New State
    tree           = state.get('tree')
    tree_dist      = int(np.floor(tree.get('dist') / dist_factor))
    tree_bot       = int(np.floor(tree.get('bot') /(tree_factor/2)))
    monkey         = state.get('monkey')
    monkey_bot     = int(np.floor(monkey.get('bot') / scale_factor))
    monkey_vel     = int(np.floor(monkey.get('vel') /vel_factor))+3

    max_val = np.max([self.q[tree_dist, tree_bot, monkey_bot, monkey_vel, :]])


    opt_action = (np.argmax([self.q[tree_dist, tree_bot, monkey_bot, monkey_vel, :]]) == True)
    new_action = opt_action
    # eps_action = (np.random.randint(2) == True)
    # test = np.random.rand()
    # if test < self.eps:
    #   new_action = eps_action
    #   # print(test)
    #   print('random!')


    new_state  = state
    a = int(new_action)

    self.q[old_tree_dist, old_tree_bot, old_monkey_bot, old_monkey_vel, a] += \
        self.rate[old_tree_dist, old_tree_bot, old_monkey_bot, old_monkey_vel, a]*((self.last_reward + self.gamma * max_val)  \
                   - self.q[old_tree_dist, old_tree_bot, old_monkey_bot, old_monkey_vel, a])

    if state.get('score') > self.hscore:
      self.hscore = state.get('score')

    self.rate[old_tree_dist, old_tree_bot, old_monkey_bot, old_monkey_vel, a] = \
      1/(1/self.rate[old_tree_dist, old_tree_bot, old_monkey_bot, old_monkey_vel, a] + 1)
    self.totscore   += state.get('score')-self.last_state.get('score')
    self.last_action = new_action
    self.last_state  = new_state
    return self.last_action


  def reward_callback(self, reward):
    """This gets called so you can see what reward you get."""

    self.last_reward = reward

iterations = 300
learner = Learner()
fifties = 0
for i in range(iterations):
  if i % 50 == 0:
    fifties = learner.total_score()

  average = learner.total_score()/(i+1)
  fiftyaverage = (learner.total_score() - fifties)/(i%50 + 1)

  # Make a new monkey object.
  swing = SwingyMonkey(sound=False,            # Don't play sounds.
                       text="Epoch %d" % i,    # Display the epoch on screen.
                       text2=str("Avg.: %f" % round(average,6)),
                       text3=str("Avg. (last 50): %f" % round(fiftyaverage,3)),
                       text4=str("High score: %g" % learner.high_score()),
                       tick_length=1,          # Make game ticks super fast.
                       action_callback=learner.action_callback,
                       reward_callback=learner.reward_callback)

  # Loop until you hit something.
  while swing.game_loop():
    pass

  # learner.eps_incr()
  # Reset the state of the learner.
  learner.reset()



